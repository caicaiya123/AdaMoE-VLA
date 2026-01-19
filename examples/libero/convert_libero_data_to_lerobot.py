"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the LeRobot datasets home directory
(commonly `$HF_LEROBOT_HOME` or a Hugging Face cache directory).
Running this conversion script will take approximately 30 minutes.
"""

import shutil
import os
from pathlib import Path
from typing import Sequence

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "your_hf_username/libero"  # Default output dataset id, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    # These match the common on-disk TFDS/RLDS folder names like:
    # <data_dir>/libero_10/1.0.0/*.tfrecord
    "libero_10",
    "libero_goal",
    "libero_object",
    "libero_spatial",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def _try_get_lerobot_home() -> Path | None:
    """Best-effort resolver for the LeRobot datasets home directory across versions.

    We keep this optional because LeRobot's public API has changed across releases.
    """
    # 0) Newer constants module (preferred)
    try:
        from lerobot.common import constants as _c  # type: ignore

        v = getattr(_c, "HF_LEROBOT_HOME", None)
        if v is not None:
            return Path(v)
    except Exception:
        pass

    # 1) Old constant that some versions expose
    try:
        from lerobot.common.datasets import lerobot_dataset as _ld  # type: ignore

        v = getattr(_ld, "LEROBOT_HOME", None)
        if v is not None:
            return Path(v)
    except Exception:
        pass

    # 2) Environment variables (user override)
    for key in (
        "HF_LEROBOT_HOME",
        "LEROBOT_HOME",
        "LEROBOT_DATASETS_HOME",
        "LEROBOT_DATASET_HOME",
    ):
        v = os.environ.get(key)
        if v:
            return Path(v).expanduser()

    # 3) Hugging Face cache fallback (matches common default: ~/.cache/huggingface/lerobot)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "lerobot"
    return Path("~/.cache/huggingface/lerobot").expanduser()


def main(
    data_dir: str,
    *,
    repo_id: str = REPO_NAME,
    raw_dataset_names: Sequence[str] = tuple(RAW_DATASET_NAMES),
    image_size: int = 224,
    overwrite: bool = False,
    push_to_hub: bool = False,
):
    # Clean up any existing dataset in the output directory
    lerobot_home = _try_get_lerobot_home()
    output_path = lerobot_home / repo_id
    if output_path.exists():
        if overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"Output dataset path already exists: {output_path}. "
                "Pass --overwrite to delete it automatically, or choose a new --repo-id."
            )
    print(f"[info] LeRobot home: {lerobot_home}")
    print(f"[info] Output dataset path: {output_path}")

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (image_size, image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],
            }
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in raw_dataset_names:
        try:
            raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load TFDS dataset '{raw_dataset_name}' from data_dir='{data_dir}'. "
                f"Expected a TFDS-style folder like '{data_dir}/{raw_dataset_name}/<version>/'. "
                f"Original error: {e}"
            ) from e
        for episode in raw_dataset:
            task: str | None = None
            for step in episode["steps"].as_numpy_iterator():
                if task is None:
                    # RLDS Libero commonly stores language instruction per-step (constant within an episode).
                    li = step.get("language_instruction", b"")
                    task = li.decode() if isinstance(li, (bytes, bytearray)) else str(li)
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "action": step["action"],
                        # Newer LeRobot versions include `task` as a required feature.
                        "task": task,
                    }
                )
            # LeRobot API differs across versions:
            # - some accept save_episode(task=...)
            # - some accept save_episode() with no args
            try:
                dataset.save_episode(task=task or "")
            except TypeError:
                dataset.save_episode()

    # 1. 确保所有图像写入完成
    dataset.stop_image_writer()

    # 2. 旧版 lerobot 不需要 consolidate，但确保 episode 缓冲区清空
    # （实际上 save_episode 已经写入，这里主要是防御性编程）
    if hasattr(dataset, "clear_episode_buffer"):
        dataset.clear_episode_buffer()

    print("[info] Dataset conversion completed.")

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
