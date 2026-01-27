#!/usr/bin/env python3
"""
Evaluate one checkpoint across all LIBERO-plus suites and perturbation categories.

This script starts the policy server (unless --reuse-server is set) and sequentially
runs `examples/libero_plus/main.py` for every suite/category combination so you get:
- Per-episode videos in a dedicated directory per category.
- Per-category results JSON written by main.py.
- A consolidated summary JSON with success rates per task.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import tyro

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_CATEGORIES = [
    "Objects Layout",
    "Camera Viewpoints",
    "Robot Initial States",
    "Language Instructions",
    "Light Conditions",
    "Background Textures",
    "Sensor Noise",
]

# DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_long"]
DEFAULT_SUITES = ["libero_object"]


def _slug(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


def _checkpoint_slug(policy: "PolicyArgs") -> str:
    """Derive a safe dir name from the checkpoint path (e.g., checkpoints/pi05_libero -> pi05_libero)."""
    raw = str(policy.checkpoint_dir).rstrip("/\\")
    leaf = raw.split("/")[-1].split("\\")[-1]
    safe = _slug(leaf)
    return safe or "checkpoint"


@dataclasses.dataclass
class PolicyArgs:
    """Policy server options (mirrors scripts/serve_policy.py)."""

    python: Optional[str] = None  # Optional override for policy server Python (separate venv)
    config: str = "pi05_libero"
    checkpoint_dir: str = "checkpoints/pi05_libero"
    host: str = "0.0.0.0"
    port: int = 8000
    default_prompt: Optional[str] = None
    clear_ld_library_path: bool = False
    cuda_visible_devices: Optional[str] = None


@dataclasses.dataclass
class EvalArgs:
    """Evaluation grid over suites x perturbation categories."""

    python: Optional[str] = None  # Optional override for eval client Python (separate venv)
    pythonpath: Optional[str] = None  # Optional PYTHONPATH to prepend for eval client
    suites: List[str] = dataclasses.field(default_factory=lambda: list(DEFAULT_SUITES))
    categories: List[str] = dataclasses.field(default_factory=lambda: list(DEFAULT_CATEGORIES))
    num_trials_per_task: int = 1
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    seed: int = 7
    video_root: str = "data/libero_plus/videos"
    results_root: str = "out/libero_plus"
    classification_path: Optional[str] = None
    mujoco_gl: str = "glx"
    use_xvfb: bool = False
    xvfb_args: List[str] = dataclasses.field(default_factory=lambda: ["-a"])
    allow_missing_classification: bool = False


@dataclasses.dataclass
class Args:
    policy: PolicyArgs = dataclasses.field(default_factory=PolicyArgs)
    eval: EvalArgs = dataclasses.field(default_factory=EvalArgs)
    reuse_server: bool = False  # Set to True if you already started serve_policy.py
    server_ready_timeout_s: float = 120.0


def _resolve_python(python_override: Optional[str]) -> str:
    """Pick an explicit python if provided, else fall back to current interpreter."""
    return str(pathlib.Path(python_override).expanduser()) if python_override else sys.executable


def _client_host(host: str) -> str:
    """Prefer a concrete loopback host over 0.0.0.0 for client connections."""
    return "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host


def _wait_for_port(host: str, port: int, timeout: float) -> bool:
    client_host = _client_host(host)
    health_url = f"http://{client_host}:{port}/healthz"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            try:
                with socket.create_connection((client_host, port), timeout=1.0):
                    return True
            except OSError:
                time.sleep(0.5)
    return False


def _start_policy_server(policy: PolicyArgs) -> subprocess.Popen:
    env = os.environ.copy()
    if policy.clear_ld_library_path:
        env.pop("LD_LIBRARY_PATH", None)
    if policy.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = policy.cuda_visible_devices

    cmd = [
        _resolve_python(policy.python),
        str(BASE_DIR / "scripts" / "serve_policy.py"),
        "--env",
        "LIBERO",
        "--port",
        str(policy.port),
    ]
    if policy.default_prompt is not None:
        cmd += ["--default-prompt", policy.default_prompt]
    cmd += [
        "policy:checkpoint",
        "--policy.config",
        policy.config,
        "--policy.dir",
        policy.checkpoint_dir,
    ]
    return subprocess.Popen(cmd, cwd=BASE_DIR, env=env)


def _stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _resolve_classification_path(eval_args: EvalArgs) -> Optional[pathlib.Path]:
    if eval_args.classification_path:
        candidate = pathlib.Path(eval_args.classification_path).expanduser()
    else:
        candidate = (
            BASE_DIR
            / "third_party"
            / "LIBERO-plus"
            / "libero"
            / "libero"
            / "benchmark"
            / "task_classification.json"
        )

    if candidate.exists():
        return candidate
    if eval_args.allow_missing_classification:
        return None
    raise FileNotFoundError(
        f"Task classification JSON not found at {candidate}. "
        "Set --eval.classification-path or pass --eval.allow-missing-classification to bypass."
    )


def _results_path_for_category(results_base: pathlib.Path, category: str) -> pathlib.Path:
    safe_category = _slug(category)
    return results_base.parent / f"{results_base.stem}_{safe_category}{results_base.suffix}"


def _build_eval_command(
    policy: PolicyArgs,
    eval_args: EvalArgs,
    suite: str,
    category: str,
    classification_path: Optional[pathlib.Path],
) -> Tuple[List[str], pathlib.Path]:
    safe_category = _slug(category)
    ckpt_slug = _checkpoint_slug(policy)
    video_dir = (BASE_DIR / eval_args.video_root / ckpt_slug / suite / safe_category).resolve()
    results_base = (BASE_DIR / eval_args.results_root / ckpt_slug / suite / "results.json").resolve()
    results_base.parent.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    connect_host = _client_host(policy.host)

    cmd: List[str] = [
        _resolve_python(eval_args.python),
        str(BASE_DIR / "examples" / "libero_plus" / "main.py"),
        "--args.task-suite-name",
        suite,
        "--args.category",
        category,
        "--args.video_out_path",
        str(video_dir),
        "--args.results-json-path",
        str(results_base),
        "--args.num_trials_per_task",
        str(eval_args.num_trials_per_task),
        "--args.resize_size",
        str(eval_args.resize_size),
        "--args.replan_steps",
        str(eval_args.replan_steps),
        "--args.num_steps_wait",
        str(eval_args.num_steps_wait),
        "--args.seed",
        str(eval_args.seed),
        "--args.host",
        connect_host,
        "--args.port",
        str(policy.port),
    ]
    if classification_path is not None:
        cmd += ["--args.task-classification-path", str(classification_path)]

    return cmd, _results_path_for_category(results_base, category)


def _run_eval_job(
    policy: PolicyArgs,
    eval_args: EvalArgs,
    suite: str,
    category: str,
    classification_path: Optional[pathlib.Path],
) -> pathlib.Path:
    cmd, results_path = _build_eval_command(policy, eval_args, suite, category, classification_path)
    env = os.environ.copy()
    # env.setdefault("MUJOCO_GL", eval_args.mujoco_gl)
    if policy.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = policy.cuda_visible_devices
    if eval_args.pythonpath:
        env["PYTHONPATH"] = (
            f"{eval_args.pythonpath}:{env['PYTHONPATH']}" if "PYTHONPATH" in env else eval_args.pythonpath
        )

    if eval_args.use_xvfb:
        cmd = ["xvfb-run", *eval_args.xvfb_args, *cmd]

    subprocess.run(cmd, cwd=BASE_DIR, env=env, check=True)
    return results_path


def _summarize_results(results_path: pathlib.Path) -> Dict:
    if not results_path.exists():
        return {}

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    counts: Dict[Tuple[int, str], Dict[str, int]] = {}

    def _accumulate(bucket: str, is_success: bool) -> None:
        for entry in data.get(bucket, []):
            key = (int(entry.get("task_id", -1)), str(entry.get("task_description", "")))
            stats = counts.setdefault(key, {"success": 0, "total": 0})
            stats["total"] += 1
            if is_success:
                stats["success"] += 1

    _accumulate("success", True)
    _accumulate("failure", False)

    per_task = []
    total_success = 0
    total_episodes = 0
    for (task_id, desc), stats in sorted(counts.items(), key=lambda x: x[0][0]):
        succ = stats["success"]
        tot = stats["total"]
        total_success += succ
        total_episodes += tot
        rate = float(succ) / float(tot) if tot > 0 else 0.0
        per_task.append(
            {
                "task_id": task_id,
                "task_description": desc,
                "successes": succ,
                "total": tot,
                "success_rate": rate,
            }
        )

    overall_rate = float(total_success) / float(total_episodes) if total_episodes > 0 else 0.0
    return {
        "overall_successes": total_success,
        "overall_episodes": total_episodes,
        "overall_success_rate": overall_rate,
        "per_task": per_task,
    }


def _write_summary(summaries: List[Tuple[str, str, pathlib.Path, Dict]], eval_args: EvalArgs) -> None:
    """Write a consolidated summary JSON for easy downstream consumption."""
    out: Dict[str, Dict[str, Dict]] = {}
    for suite, category, results_path, summary in summaries:
        out.setdefault(suite, {})[category] = {
            "results_path": str(results_path),
            **summary,
        }

    if summaries:
        # All summaries share the same root: .../results_root/<ckpt_slug>/<suite>/
        summary_root = pathlib.Path(summaries[0][2]).parent.parent
    else:
        summary_root = (BASE_DIR / eval_args.results_root).resolve()
    summary_path = (summary_root / "summary_all.json").resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nWrote consolidated summary to {summary_path}")


def main(args: Args) -> None:
    # Allow server (policy) and client (libero eval) to use different virtualenvs.
    args.policy.python = _resolve_python(args.policy.python)
    args.eval.python = _resolve_python(args.eval.python)
    print(f"Using policy server python: {args.policy.python}")
    print(f"Using eval client python: {args.eval.python}")
    if args.eval.pythonpath:
        print(f"Prepended to eval PYTHONPATH: {args.eval.pythonpath}")

    classification_path = _resolve_classification_path(args.eval)

    server_proc: Optional[subprocess.Popen] = None
    if args.reuse_server:
        if not _wait_for_port(args.policy.host, args.policy.port, args.server_ready_timeout_s):
            raise RuntimeError(
                f"Policy server not reachable at {args.policy.host}:{args.policy.port} (reuse_server=True)."
            )
    else:
        server_proc = _start_policy_server(args.policy)
        if not _wait_for_port(args.policy.host, args.policy.port, args.server_ready_timeout_s):
            _stop_process(server_proc)
            raise RuntimeError("Policy server failed to start or did not open the port in time.")

    summaries: List[Tuple[str, str, pathlib.Path, Dict]] = []
    try:
        for suite in args.eval.suites:
            for category in args.eval.categories:
                print(f"\n==== Evaluating suite={suite}, category={category} ====")
                results_path = _run_eval_job(args.policy, args.eval, suite, category, classification_path)
                summary = _summarize_results(results_path)
                summaries.append((suite, category, results_path, summary))

                overall = summary.get("overall_success_rate", 0.0)
                succ = summary.get("overall_successes", 0)
                total = summary.get("overall_episodes", 0)
                print(f"[{suite} | {category}] success_rate={overall:.3f} ({succ}/{total}) -> {results_path}")

                for task in summary.get("per_task", []):
                    rate = task["success_rate"]
                    print(
                        f"  task {task['task_id']:02d}: {rate:.3f} "
                        f"({task['successes']}/{task['total']}) | {task['task_description']}"
                    )
    finally:
        if server_proc is not None:
            _stop_process(server_proc)

    _write_summary(summaries, args.eval)


if __name__ == "__main__":
    tyro.cli(main)
