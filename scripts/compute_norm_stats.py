"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro
import math
import torch
import dataclasses

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

class RemoveStrings(transforms.DataTransformFn):

    def __call__(self, x: dict) -> dict:
        return {
            k: v
            for k, v in x.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }

@dataclasses.dataclass(frozen=True)
class ExtractStateActions(transforms.DataTransformFn):
    """Extract only numeric columns needed for norm stats.

    This avoids running expensive image/video transforms during stats computation.
    """

    action_dim: int

    def __call__(self, x: dict) -> dict:
        # Different LeRobot datasets expose state under different key conventions.
        # We keep this intentionally small and explicit.
        if "observation.state" in x:
            state = x["observation.state"]
        elif "observation/state" in x:
            state = x["observation/state"]
        elif "state" in x:
            state = x["state"]
        else:
            raise KeyError(
                'Missing state in dataset sample. Tried keys: '
                '"observation.state", "observation/state", "state". '
                f"Available keys (sample): {list(x.keys())[:30]}"
            )
        state = np.asarray(state)

        if "action" in x:
            actions = x["action"]
        elif "actions" in x:
            actions = x["actions"]
        else:
            raise KeyError('Missing "action"/"actions" in dataset sample')
        actions = np.asarray(actions)

        # Match training-time shapes: LiberoInputs pads state/actions to model action_dim (Pi0 default: 32).
        # - state: (8,) -> (action_dim,)
        # - actions: (H,7) -> (H,action_dim) (H = action_horizon)
        state = transforms.pad_to_dim(state, self.action_dim)
        actions = transforms.pad_to_dim(actions, self.action_dim)
        return {"state": state, "actions": actions}


def _drop_expensive_columns_if_possible(dataset) -> None:
    """Best-effort column dropping to avoid decoding videos/images in `datasets`."""
    hf = getattr(dataset, "hf_dataset", None)
    if hf is None or not hasattr(hf, "column_names") or not hasattr(hf, "remove_columns"):
        return

    cols = list(hf.column_names)
    drop = [
        c
        for c in cols
        if c.startswith("observation.images.")
        or c in {"observation/image", "observation/wrist_image", "image", "wrist_image"}
    ]
    if drop:
        dataset.hf_dataset = hf.remove_columns(drop)


def create_dataset(
    config: _config.TrainConfig,
) -> tuple[_config.DataConfig, _data_loader.Dataset]:
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_dataset(data_config, config.model)
    _drop_expensive_columns_if_possible(dataset)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            # Only keep numeric columns needed for stats (avoid image/video decoding).
            ExtractStateActions(action_dim=config.model.action_dim),
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    return data_config, dataset


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config, dataset = create_dataset(config)

    # NOTE:
    # - `len(dataset)` is number of frames/examples, not number of batches.
    # - We compute stats over the entire batch (not just batch[0]).
    # - We avoid the OpenPI TorchDataLoader wrapper (which converts to JAX arrays),
    #   because for norm stats we only need numpy.
    num_frames_total = len(dataset)
    target_frames = min(max_frames, num_frames_total) if max_frames is not None else num_frames_total

    batch_size = 4096  # safe: only reading small numeric tensors (state/actions)
    num_workers = 8
    num_batches = math.ceil(target_frames / batch_size)

    torch_loader = torch.utils.data.DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=batch_size,
        shuffle=(max_frames is not None and max_frames < num_frames_total),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    seen = 0
    for batch in tqdm.tqdm(torch_loader, total=num_batches, desc="Computing stats"):
        # batch[key] is typically a torch Tensor after default collate.
        bsz = None
        for key in keys:
            values = batch[key]
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            else:
                values = np.asarray(values)
            if bsz is None:
                bsz = values.shape[0]
            stats[key].update(values.reshape(-1, values.shape[-1]))
        seen += int(bsz or 0)
        if seen >= target_frames:
            break

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
