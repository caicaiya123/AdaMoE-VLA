from collections.abc import Iterator, Sequence
import multiprocessing
import os
import pathlib
import inspect
import typing
from typing import Protocol, SupportsIndex, TypeVar
import math

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

from lerobot.common.datasets.utils import load_info

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError(
            "Subclasses of DataLoader should implement data_config."
        )

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):

    def __init__(
        self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):

    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(
                    data_rng, shape=shape, minval=-1.0, maxval=1.0
                )
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_dataset(
    data_config: _config.DataConfig, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # Local dataset support:
    # - If data_config.local_root_dir is provided, treat it as the actual dataset directory on disk.
    #   We adapt by setting HF_LEROBOT_HOME to its parent and using repo_id=dir.name unless explicitly set.
    # - If repo_id itself is an absolute path, treat it as local_root_dir for convenience.
    local_root_dir: pathlib.Path | None = None
    if isinstance(repo_id, str) and repo_id.startswith("/"):
        local_root_dir = pathlib.Path(repo_id).expanduser().resolve()
        repo_id = local_root_dir.name
    elif data_config.local_root_dir is not None:
        local_root_dir = pathlib.Path(data_config.local_root_dir).expanduser().resolve()
        if not local_root_dir.exists():
            raise FileNotFoundError(f"Local LeRobot dataset directory not found: {local_root_dir}")
        # If the user kept repo_id as a HF-style name, we won't try to rewrite it.
        # If they want to load exactly from local_root_dir, they can either:
        # - set repo_id to local_root_dir.name, or
        # - set repo_id to the absolute path (handled above).
        if repo_id in [None, "", "physical-intelligence/libero"]:
            repo_id = local_root_dir.name

    if local_root_dir is not None:
        # LeRobot uses HF_LEROBOT_HOME / <repo_id> as the on-disk layout. We set the env var
        # so that LeRobotDataset(repo_id) resolves to the provided local_root_dir.
        # Set HF_LEROBOT_HOME to the parent directory of local_root_dir
        hf_lerobot_home = str(local_root_dir.parent)
        os.environ["HF_LEROBOT_HOME"] = hf_lerobot_home
        
        # Verify that the expected path structure matches local_root_dir
        # LeRobotDataset expects: $HF_LEROBOT_HOME/<repo_id>
        expected_path = pathlib.Path(hf_lerobot_home) / repo_id
        if not expected_path.resolve().exists():
            raise FileNotFoundError(
                f"LeRobot dataset directory not found at expected path: {expected_path}. "
                f"local_root_dir is: {local_root_dir}, "
                f"HF_LEROBOT_HOME is: {hf_lerobot_home}, repo_id is: {repo_id}. "
                f"Please ensure the dataset directory exists at {expected_path}."
            )
        
        # Enforce local-only if requested: validate expected directory exists.
        if data_config.local_files_only:
            if not local_root_dir.exists():
                raise FileNotFoundError(
                    f"local_files_only=True but LeRobot dataset was not found on disk. "
                    f"Expected at: {local_root_dir}. "
                    f"HF_LEROBOT_HOME is set to: {hf_lerobot_home}, repo_id is: {repo_id}"
                )
        
        actual_root = local_root_dir
    else:
        # 默认行为（不推荐用于本地）
        actual_root = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser() / repo_id

    # 直接从本地加载 info.json
    info = load_info(actual_root)
    fps = info["fps"]
    # 如果你需要 tasks 列表（用于 prompt_from_task）：
    tasks = info.get("tasks", [])  # 注意：旧版转换脚本可能没写 tasks

    # 构造一个 mock metadata 对象
    class MockDatasetMeta:
        def __init__(self, fps, tasks):
            self.fps = fps
            self.tasks = tasks

    dataset_meta = MockDatasetMeta(fps=fps, tasks=tasks)

    # If we are explicitly configured to use local files only, proactively validate
    # that the expected parquet layout exists. This avoids lerobot falling back to
    # Hugging Face Hub when it detects missing local files.
    if data_config.local_files_only and local_root_dir is not None:
        try:
            total_episodes = int(info["total_episodes"])
            chunks_size = int(info.get("chunks_size", 1000))
            data_path_tmpl = info["data_path"]
        except Exception as e:
            raise ValueError(
                f"Invalid LeRobot info.json under {actual_root}/meta/info.json; missing required fields. "
                f"Got keys: {list(info.keys())}"
            ) from e

        num_chunks = math.ceil(total_episodes / max(1, chunks_size))
        missing: list[pathlib.Path] = []
        for chunk in range(num_chunks):
            first_ep = chunk * chunks_size
            last_ep = min(total_episodes - 1, chunk * chunks_size + chunks_size - 1)
            for ep in (first_ep, last_ep):
                rel = data_path_tmpl.format(episode_chunk=chunk, episode_index=ep)
                p = actual_root / rel
                if not p.is_file():
                    missing.append(p)
                    if len(missing) >= 5:
                        break
            if len(missing) >= 5:
                break

        if missing:
            sample = "\n".join(f"- {p}" for p in missing)
            raise FileNotFoundError(
                "Local LeRobot dataset appears incomplete or has a naming/layout mismatch.\n"
                f"Dataset root: {actual_root}\n"
                f"info.json data_path template: {data_path_tmpl}\n"
                "Some expected parquet files are missing (sample):\n"
                f"{sample}\n"
                "Common causes:\n"
                "- chunk directory names differ (e.g. chunk-0 vs chunk-000)\n"
                "- episode file names differ (e.g. episode_0.parquet vs episode_000000.parquet)\n"
                "- partial download/copy: some chunks missing\n"
            )

    # Compatibility across lerobot versions:
    # - Some versions accept LeRobotDataset(..., local_files_only=...).
    # - Others don't; in that case we still enforce local-only via an env var and
    #   our explicit on-disk path checks above.
    lerobot_kwargs: dict[str, typing.Any] = {
        "delta_timestamps": {
            key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
            for key in data_config.action_sequence_keys
        }
    }
    local_only = bool(data_config.local_files_only and local_root_dir is not None)
    if local_only:
        try:
            sig = inspect.signature(lerobot_dataset.LeRobotDataset)
            if "local_files_only" in sig.parameters:
                lerobot_kwargs["local_files_only"] = True
            else:
                # huggingface_hub respects this and will avoid network calls.
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
        except (TypeError, ValueError):
            # If signature introspection fails for any reason, fall back to offline mode.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

    dataset = lerobot_dataset.LeRobotDataset(repo_id, **lerobot_kwargs)

    if data_config.prompt_from_task:
        dataset = TransformedDataset(
            dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)]
        )

    return dataset


def transform_dataset(
    dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False
) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    data_config = config.data.create(config.assets_dirs, config.model)

    dataset = create_dataset(data_config, config.model)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=config.seed,
    )

    class DataLoaderImpl(DataLoader):

        def __init__(
            self, data_config: _config.DataConfig, data_loader: TorchDataLoader
        ):
            self._data_config = data_config
            self._data_loader = data_loader

        def data_config(self) -> _config.DataConfig:
            return self._data_config

        def __iter__(self):
            for batch in self._data_loader:
                yield _model.Observation.from_dict(batch), batch["actions"]

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError(
                "Data loading with multiple processes is not supported."
            )

        if len(dataset) < local_batch_size:
            raise ValueError(
                f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)})."
            )

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(self._sharding, x),
                    batch,
                )


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
