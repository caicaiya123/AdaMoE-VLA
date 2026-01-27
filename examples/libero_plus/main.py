import collections
import dataclasses
import logging
import math
import pathlib
from typing import Optional, List, Set, Dict, Any, Tuple
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

# ---- NEW: imports for results JSON logging ----
import json
import os
import datetime as dt
import traceback

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90,
    # or "all" to evaluate across all suites
    task_suite_name: str = "libero_spatial"
    # Optional category filter (from LIBERO-plus task_classification.json). When set, only
    # tasks in the selected suite(s) with matching category are evaluated. Example: "Camera Viewpoints"
    category: Optional[str] = None
    # Optional override path for LIBERO-plus classification JSON
    task_classification_path: Optional[str] = None
    task_ids: Optional[str] = None  # Examples: "123" or "100-199" or "0,7,10-12".
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    #################################################################################################################
    # ---- NEW: Results JSON output ----
    #################################################################################################################
    # Where to write the rolling results JSON. Configure via:
    #   --args.results-json-path path/to/results.json
    # If category is set, will auto-append category name: results_{category}.json
    results_json_path: str = "data/libero/results.json"


def _parse_task_ids(expr: Optional[str], upper: int) -> List[int]:
    """
    Parse expressions like "5", "10-20", "0,7,10-12" into a sorted, de-duplicated
    list of integer task ids within [0, upper-1]. Ranges are inclusive.
    """
    if expr is None or str(expr).strip() == "":
        return list(range(upper))
    s = str(expr).replace(" ", "")
    out: Set[int] = set()
    parts = [p for p in s.split(",") if p != ""]
    for part in parts:
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start = int(a); end = int(b)
            except ValueError:
                raise ValueError(f'Invalid range "{part}" in --args.task-ids.')
            if start > end:
                start, end = end, start
            for i in range(start, end + 1):
                if 0 <= i < upper:
                    out.add(i)
                else:
                    raise ValueError(
                        f"Task id {i} out of range [0, {upper-1}] for suite (from range {part})."
                    )
        else:
            try:
                i = int(part)
            except ValueError:
                raise ValueError(f'Invalid id "{part}" in --args.task-ids.')
            if 0 <= i < upper:
                out.add(i)
            else:
                raise ValueError(f"Task id {i} out of range [0, {upper-1}] for suite.")
    return sorted(out)


# ---- NEW: small helpers for safe, real-time JSON logging -------------------------------------
def _atomic_write_json(obj: Dict[str, Any], path: pathlib.Path) -> None:
    """Write JSON atomically: write to tmp then replace to avoid partial files."""
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file in same directory (important for atomic rename)
    tmp_path = path.parent / f".{path.name}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        # Atomic replace: works across platforms
        if os.path.exists(path):
            os.replace(tmp_path, path)
        else:
            # First time: ensure directory exists then rename
            tmp_path.rename(path)
    except Exception as e:
        # Cleanup temp file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise e


def _init_results_file(args: Args, selected) -> pathlib.Path:
    """Create or update the results file header (meta + empty buckets)."""
    path = pathlib.Path(args.results_json_path)
    now_iso = dt.datetime.now().isoformat()
    data: Dict[str, Any]
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    data.setdefault("meta", {})
    meta = data["meta"]
    # Keep original created_at if present, otherwise set it.
    meta.setdefault("created_at", now_iso)
    # Always update these
    meta.update(
        {
            "updated_at": now_iso,
            "task_suite_name": args.task_suite_name,
            # Can be a list of ids for one suite or a map of suite->ids
            "selected_task_ids": selected,
            "host": args.host,
            "port": args.port,
            "resize_size": args.resize_size,
            "replan_steps": args.replan_steps,
            "num_trials_per_task": args.num_trials_per_task,
            "seed": args.seed,
            "video_out_path": str(args.video_out_path),
        }
    )

    data.setdefault("success", [])
    data.setdefault("failure", [])
    data.setdefault(
        "running_counts",
        {"total_episodes": 0, "total_successes": 0, "success_rate": 0.0},
    )
    _atomic_write_json(data, path)
    return path


def _is_episode_completed(args: Args, task_description: str, episode_index: int) -> Tuple[bool, Optional[bool]]:
    """
    Check if an episode has already been completed by checking video file existence.
    
    Returns:
        Tuple: (is_completed: bool, was_success: Optional[bool])
            - is_completed: True if video file exists (either success or failure)
            - was_success: True if success, False if failure, None if not completed
    """
    task_segment = task_description.replace(" ", "_")
    video_dir = pathlib.Path(args.video_out_path)
    
    # Check for success video
    success_video = video_dir / f"rollout_{task_segment}_ep{episode_index:02d}_success.mp4"
    if success_video.exists():
        return True, True
    
    # Check for failure video
    failure_video = video_dir / f"rollout_{task_segment}_ep{episode_index:02d}_failure.mp4"
    if failure_video.exists():
        return True, False
    
    return False, None


def _record_episode_result(
    args: Args,
    *,
    task_id: int,
    task_description: str,
    episode_index: int,
    steps_taken: int,
    success: bool,
    video_path: pathlib.Path,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one episode result and persist to disk immediately."""
    path = pathlib.Path(args.results_json_path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {
            "meta": {},
            "success": [],
            "failure": [],
            "running_counts": {
                "total_episodes": 0,
                "total_successes": 0,
                "success_rate": 0.0,
            },
        }

    # Check if this episode is already recorded (避免重复记录)
    bucket = "success" if success else "failure"
    for existing_record in data.get(bucket, []):
        if (existing_record.get("task_id") == task_id and 
            existing_record.get("episode_index") == episode_index and
            existing_record.get("task_description") == task_description):
            # Already recorded, skip
            return

    record: Dict[str, Any] = {
        "timestamp": dt.datetime.now().isoformat(),
        "task_id": int(task_id),
        "task_description": str(task_description),
        "episode_index": int(episode_index),
        "steps_taken": int(steps_taken),
        "video": str(video_path),
    }
    if error:
        record["error"] = str(error)
    if extra:
        record["extra"] = extra

    data.setdefault(bucket, []).append(record)

    rc = data.setdefault(
        "running_counts",
        {"total_episodes": 0, "total_successes": 0, "success_rate": 0.0},
    )
    rc["total_episodes"] = int(rc.get("total_episodes", 0)) + 1
    if success:
        rc["total_successes"] = int(rc.get("total_successes", 0)) + 1
    total = max(1, rc["total_episodes"])
    rc["success_rate"] = float(rc["total_successes"]) / float(total)

    # touch meta.updated_at
    data.setdefault("meta", {})
    data["meta"]["updated_at"] = dt.datetime.now().isoformat()

    _atomic_write_json(data, path)


def eval_libero(args: Args) -> None:
    # Auto-adjust results file path based on category to avoid conflicts
    if args.category is not None:
        base_path = pathlib.Path(args.results_json_path)
        safe_category = args.category.replace(" ", "_").replace("/", "_")
        args.results_json_path = str(base_path.parent / f"{base_path.stem}_{safe_category}{base_path.suffix}")
        logging.info(f"Category-specific results will be saved to: {args.results_json_path}")
    
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite(s)
    benchmark_dict = benchmark.get_benchmark_dict()
    if args.task_suite_name == "all":
        suite_names = [
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        ]
    else:
        suite_names = [args.task_suite_name]

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # Optional category filtering via LIBERO-plus classification JSON
    classification_by_suite: Dict[str, Set[str]] = {}
    if args.category is not None:
        classification_path = (
            pathlib.Path(args.task_classification_path).expanduser()
            if args.task_classification_path
            else pathlib.Path(__file__).resolve().parents[3]
            / "LIBERO-plus"
            / "libero"
            / "libero"
            / "benchmark"
            / "task_classification.json"
        )
        try:
            with open(classification_path, "r", encoding="utf-8") as f:
                classification = json.load(f)
            for s in suite_names:
                if s in classification:
                    names = {
                        entry["name"]
                        for entry in classification[s]
                        if entry.get("category") == args.category
                    }
                    classification_by_suite[s] = names
                else:
                    classification_by_suite[s] = set()
            logging.info(
                f"Category filter enabled: '{args.category}'. Using classification at {classification_path}"
            )
        except Exception as e:
            logging.warning(
                f"Failed to load task classification from {classification_path}: {e}. Proceeding without category filter."
            )
            classification_by_suite = {}

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # ---- NEW: initialize results file with meta header (per-suite map of selected ids) ----
    selected_map: Dict[str, List[int]] = {}

    # Global counters across suites
    total_episodes, total_successes = 0, 0

    for suite_name in suite_names:
        key = suite_name.lower()
        if key not in benchmark_dict:
            available = sorted(list(benchmark_dict.keys()))
            raise ValueError(
                f"Unknown task suite: {suite_name}. "
                f"Available suites in your installed `libero` are: {available}. "
                "Note: Some LIBERO versions use suites like `libero_10`/`libero_90` instead of `libero_long`. "
                "Double-check (1) your `PYTHONPATH` points to the intended LIBERO/LIBERO-plus checkout, "
                "and (2) the suite names you pass to the evaluator match `benchmark.get_benchmark_dict()`."
            )

        task_suite = benchmark_dict[key]()
        num_tasks_in_suite = task_suite.n_tasks
        if suite_name == "libero_spatial":
            max_steps = 220  # longest training demo has 193 steps
        elif suite_name == "libero_object":
            max_steps = 280  # longest training demo has 254 steps
        elif suite_name == "libero_goal":
            max_steps = 300  # longest training demo has 270 steps
        elif suite_name == "libero_10":
            max_steps = 520  # longest training demo has 505 steps
        elif suite_name == "libero_90":
            max_steps = 400  # longest training demo has 373 steps
        else:
            raise ValueError(f"Unknown task suite: {suite_name}")

        # Determine which task ids to run for this suite
        selected_task_ids = _parse_task_ids(args.task_ids, num_tasks_in_suite)
        allowed_task_names = classification_by_suite.get(suite_name)

        # NEW: pre-filter ids by category (if provided)
        if allowed_task_names is not None:
            filtered_task_ids = [
                tid for tid in selected_task_ids
                if task_suite.get_task(tid).name in allowed_task_names
            ]
        else:
            filtered_task_ids = selected_task_ids

        # Keep what we actually plan to run in the results header
        selected_map[suite_name] = filtered_task_ids

        logging.info(
            f"Evaluating suite: {suite_name} | tasks: {num_tasks_in_suite} | "
            f"category: {args.category or 'ALL'} | selected ids: {filtered_task_ids[:10]}"
            f"{' ...' if len(filtered_task_ids) > 10 else ''}"
        )
        logging.info(f"[{suite_name}] category matched: {len(filtered_task_ids)}/{len(selected_task_ids)}")

        # Initialize / update results header once with per-suite selection map
        _init_results_file(args, selected_map)

        # Start evaluation for this suite
        for task_id in tqdm.tqdm(filtered_task_ids, total=len(filtered_task_ids)):
            # Get task
            task = task_suite.get_task(task_id)

            # Get default LIBERO initial states
            initial_states = task_suite.get_task_init_states(task_id)

            # Initialize LIBERO environment and task description
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

            # Start episodes
            task_episodes, task_successes = 0, 0
            for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"episodes for task {task_id}"):
                # ---- RESUME: Skip if already completed (check video file) ----
                is_completed, was_success = _is_episode_completed(args, task_description, episode_idx)
                if is_completed:
                    status = "SUCCESS" if was_success else "FAILURE"
                    logging.info(f"⏭️  Skipping task {task_id} episode {episode_idx}: already completed ({status})")
                    task_episodes += 1
                    total_episodes += 1
                    if was_success:
                        task_successes += 1
                        total_successes += 1
                    
                    # Ensure JSON record exists (补录已完成但未记录的任务)
                    suffix = "success" if was_success else "failure"
                    task_segment = task_description.replace(" ", "_")
                    video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_ep{episode_idx:02d}_{suffix}.mp4"
                    _record_episode_result(
                        args,
                        task_id=task_id,
                        task_description=task_description,
                        episode_index=episode_idx,
                        steps_taken=-1,  # Unknown for skipped episodes
                        success=was_success,
                        video_path=video_path,
                        error=None,
                        extra={"max_steps": max_steps, "num_steps_wait": args.num_steps_wait, "suite": suite_name, "skipped": True},
                    )
                    continue
                
                logging.info(f"\nTask: {task_description} | episode {episode_idx+1}/{args.num_trials_per_task}")

                ep_seed = int(args.seed + 1000 * int(task_id) + int(episode_idx))
                env.seed(ep_seed)

                # Reset & plan buffer
                env.reset()
                action_plan = collections.deque()

                if len(initial_states) == 0:
                    raise RuntimeError(f"No initial states for task {task_id}")
                rng = np.random.RandomState(ep_seed)
                init_idx = int(rng.randint(len(initial_states)))

                obs = env.set_init_state(initial_states[init_idx])

                # Setup
                t = 0
                replay_images = []
                done = False
                last_error = None

                logging.info(f"Starting episode {task_episodes+1}...")
                while t < max_steps + args.num_steps_wait:
                    try:
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                        wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))

                        replay_images.append(img)

                        if not action_plan:
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": np.concatenate(
                                    (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                                ),
                                "prompt": str(task_description),
                            }
                            action_chunk = client.infer(element)["actions"]
                            assert len(action_chunk) >= args.replan_steps, \
                                f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                            action_plan.extend(action_chunk[: args.replan_steps])

                        action = action_plan.popleft()

                        obs, reward, done, info = env.step(action.tolist())
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                    except Exception as e:
                        last_error = f"{type(e).__name__}: {e}"
                        logging.error(f"Caught exception: {e}")
                        break

                # === episode post jobs ===
                task_episodes += 1
                total_episodes += 1

                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_ep{episode_idx:02d}_{suffix}.mp4"
                imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

                logging.info(f"Success: {done}")
                logging.info(f"# episodes completed so far: {total_episodes}")
                logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

                _record_episode_result(
                    args,
                    task_id=task_id,
                    task_description=task_description,
                    episode_index=episode_idx,
                    steps_taken=t,
                    success=bool(done),
                    video_path=video_path,
                    error=last_error,
                    extra={"max_steps": max_steps, "num_steps_wait": args.num_steps_wait, "suite": suite_name},
                )

        # Log final results for this suite
        logging.info(f"[Suite {suite_name}] task success rate: {float(task_successes) / float(task_episodes) if task_episodes else 0.0}")
        logging.info(f"[Suite {suite_name}] total success rate: {float(total_successes) / float(total_episodes) if total_episodes else 0.0}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": str(task_bddl_file), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)