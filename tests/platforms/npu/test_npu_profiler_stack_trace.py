# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ascend hardware validation for the NPU profiler ``with_stack`` forwarding.

``tests/platforms/npu/test_npu_torch_profiler.py`` installs a fake
``torch_npu`` and asserts the keyword arguments that
``NPUTorchProfilerWrapper._create_profiler`` forwards. That covers the source
contract and nothing else: the module it fakes is the component the issue
reports on, so it cannot answer whether the real SDK plus the real device put
Python stacks into an exported trace.

This module is the instrument for that remaining question, and it drives the
real path (``NPUTorchProfilerWrapper`` -> ``torch_npu.profiler.profile``) in two
matched arms:

* control arm -- ``torch_profiler_with_stack=False``
* head arm    -- ``torch_profiler_with_stack=True``

Each arm exports whatever the NPU route produces and counts the Python frames
found in that artifact. The run is only accepted when the control arm stays
empty **and** the head arm carries frames. The control arm is what makes the
count mean something: without it, a frame the SDK writes unconditionally would
be indistinguishable from the feature working.

Two export schemas are counted on the union, so a schema difference is not read
as a missing feature: Kineto-style exporters emit Python frames as
``cat == "python_function"`` events named ``path/to/file.py(line): func``, while
other exporters write a ``stack`` array on the event or in ``args``.

As a test (skips on hosts without an Ascend device)::

    pytest tests/platforms/npu/test_npu_profiler_stack_trace.py -v

As a standalone instrument that leaves the matched traces plus an environment
manifest for the reviewer (run from the repo root, so ``tests`` is importable)::

    python -m tests.platforms.npu.test_npu_profiler_stack_trace --out-dir /tmp/npu-stack-8877

Exit status: ``0`` the discriminator held, ``1`` it did not, ``2`` the
instrument could not run (no device, or no trace artifact was produced).
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

try:  # a direct-path ``python <file>`` run has no package root on sys.path
    from tests.helpers.mark import hardware_marks
except ModuleNotFoundError:  # pragma: no cover - standalone instrument only
    hardware_marks = None

_MARKER = "HUELLA-INSTRUMENTO-STACK-A5-20261023"
_CONTROL_ARM = "control_with_stack_false"
_HEAD_ARM = "head_with_stack_true"
_WORKLOAD = "3 x relu(fp16 1024x1024 @ npu)"

# ``path/to/file.py(812): __enter__`` (kineto) or ``path/to/file.py:812``.
_PY_FRAME_NAME = re.compile(r"\.py\((?:line )?\d+\)|\.py:\d+")
_PY_ID_KEYS = ("Python id", "Python parent id")


def _npu_available() -> bool:
    """True only for a real device: a fake ``torch_npu`` cannot answer this."""
    npu = getattr(torch, "npu", None)
    if npu is None:
        return False
    try:
        import torch_npu  # noqa: F401

        return bool(npu.is_available())
    except Exception:
        return False


# One SKU per item, per repo convention. A3 is the Ascend SKU the CI pools are
# built on -- most ``npu`` entries under ``tests/dfx/**`` declare A3 -- so the
# probe is selected where it can actually run. The issue was reported on A5;
# ad-hoc runs there do not need the mark (the standalone CLI is SKU-agnostic),
# and the skip below still refuses to pretend on a host without the device.
pytestmark = [
    pytest.mark.local_model,
    *(hardware_marks(res={"npu": "A3"}, num_cards=1) if hardware_marks else []),
    pytest.mark.skipif(
        not _npu_available(),
        reason="requires an Ascend NPU device: this test measures the exported trace, not a mock",
    ),
]


def _workload() -> None:
    """Small real NPU workload; these are the ops the profiler has to annotate."""
    device = torch.device("npu")
    left = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    right = torch.randn(1024, 1024, device=device, dtype=torch.float16)
    for _ in range(3):
        left = torch.relu(left @ right)
    torch.npu.synchronize()


def _iter_stacks(node: Any) -> Iterator[list[Any]]:
    """Yield every non-empty ``stack`` list found anywhere in the trace structure."""
    if isinstance(node, dict):
        stack = node.get("stack")
        if isinstance(stack, list) and stack:
            yield stack
        for value in node.values():
            yield from _iter_stacks(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_stacks(item)


def _carries_python_frame(event: Any) -> bool:
    """True when one trace entry is a Python frame, in either export schema."""
    if not isinstance(event, dict):
        return False
    if event.get("cat") == "python_function":
        return True
    name = event.get("name")
    if isinstance(name, str) and _PY_FRAME_NAME.search(name):
        return True
    args = event.get("args")
    if isinstance(args, dict):
        if any(key in args for key in _PY_ID_KEYS):
            return True
        stack = args.get("stack")
        if isinstance(stack, list) and stack:
            return True
    stack = event.get("stack")
    return isinstance(stack, list) and bool(stack)


def _measure(trace_file: Path) -> dict[str, Any]:
    """Count Python frames and ``stack`` arrays in one exported trace."""
    payload = json.loads(trace_file.read_text(encoding="utf-8"))
    events = payload.get("traceEvents") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        events = []

    stack_lists = 0
    stack_frames = 0
    python_frames = 0
    samples: list[list[Any]] = []
    for event in events:
        for stack in _iter_stacks(event):
            stack_lists += 1
            stack_frames += len(stack)
            if len(samples) < 3:
                samples.append(stack[:3])
        if _carries_python_frame(event):
            python_frames += 1
            name = event.get("name") if isinstance(event, dict) else None
            if isinstance(name, str) and _PY_FRAME_NAME.search(name) and len(samples) < 3:
                samples.append([name])

    return {
        "events": len(events),
        "stack_lists": stack_lists,
        "stack_frames": stack_frames,
        "python_frames": python_frames,
        "sample_frames": samples,
    }


def _latest_trace_json(directory: Path) -> Path | None:
    """Pick the artifact the SDK wrote; ``trace_view.json`` when it exists."""
    candidates = [path for path in directory.rglob("*.json") if path.is_file()]
    if not candidates:
        return None
    preferred = [path for path in candidates if "trace" in path.name.lower()]
    return max(preferred or candidates, key=lambda path: path.stat().st_size)


def _profile_arm(out_dir: Path, *, with_stack: bool) -> dict[str, Any]:
    """Run the real NPU profiler route once and measure its artifact."""
    from vllm.config import ProfilerConfig

    from vllm_omni.platforms.npu.profiler import NPUTorchProfilerWrapper

    out_dir.mkdir(parents=True, exist_ok=True)
    config = ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=str(out_dir),
        torch_profiler_with_stack=with_stack,
        torch_profiler_use_gzip=False,
    )
    profiler = NPUTorchProfilerWrapper(
        config,
        worker_name="stack_trace_probe",
        local_rank=0,
    ).profiler

    profiler.start()
    try:
        _workload()
    finally:
        profiler.stop()
        torch.npu.synchronize()

    trace_file = _latest_trace_json(out_dir)
    measurement: dict[str, Any] = {
        "with_stack": with_stack,
        "trace_dir": str(out_dir),
        "trace_file": str(trace_file) if trace_file is not None else None,
        "events": 0,
        "stack_lists": 0,
        "stack_frames": 0,
        "python_frames": 0,
        "sample_frames": [],
    }
    if trace_file is not None:
        measurement.update(_measure(trace_file))
    return measurement


def _manifest() -> dict[str, str]:
    import torch_npu

    try:
        device_name = torch.npu.get_device_name(0)
    except Exception:
        device_name = "unknown"
    return {
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "cann": getattr(getattr(torch, "version", None), "cann", "unknown"),
        "device": device_name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "vllm_omni_commit": _git_commit(),
        "workload": _WORKLOAD,
        "probe": _MARKER,
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _verdict(arms: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    """Decide from both arms; a single-arm reading is not admissible."""
    control = arms[_CONTROL_ARM]
    head = arms[_HEAD_ARM]

    if control["python_frames"] != 0:
        return False, (
            "control arm is contaminated: with torch_profiler_with_stack=False the exported trace already carries "
            f"{control['python_frames']} python frame(s), so this count cannot discriminate the fix. Treat the "
            "instrument as invalid and compare both artifacts in the Ascend trace viewer instead."
        )
    if head["python_frames"] == 0:
        return False, (
            "head arm carries no python frame "
            f"({head['stack_lists']} stack array(s), {head['stack_frames']} frame(s) total): the flag reaches "
            "torch_npu but the exported trace still has no Python stack. The artifact-level claim of #8877 does "
            "not hold yet."
        )
    return True, (
        f"head arm carries {head['python_frames']} python frame(s) "
        f"({head['stack_lists']} stack array(s), {head['stack_frames']} frame(s) total); control arm carries none."
    )


def test_npu_trace_carries_python_stacks_only_when_requested(tmp_path: Path) -> None:
    arms = {
        _CONTROL_ARM: _profile_arm(tmp_path / "control", with_stack=False),
        _HEAD_ARM: _profile_arm(tmp_path / "head", with_stack=True),
    }
    passed, reason = _verdict(arms)
    assert passed, f"{reason}\ncontrol={arms[_CONTROL_ARM]}\nhead={arms[_HEAD_ARM]}"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched-arm validation of torch_profiler_with_stack on an Ascend NPU.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where the matched traces and manifest are written (default: a temporary directory).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not _npu_available():
        print(
            "CANNOT RUN: no Ascend NPU device. This instrument measures the real SDK and device; "
            "a mock cannot stand in for them.",
            file=sys.stderr,
        )
        return 2

    out_dir: Path = args.out_dir or Path(tempfile.mkdtemp(prefix="npu-stack-8877-"))
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _manifest()
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    arms: dict[str, dict[str, Any]] = {}
    for name, with_stack in ((_CONTROL_ARM, False), (_HEAD_ARM, True)):
        arms[name] = _profile_arm(out_dir / name, with_stack=with_stack)
        (out_dir / name / "measurement.json").write_text(json.dumps(arms[name], indent=2), encoding="utf-8")

    passed, reason = _verdict(arms)
    report = {"manifest": manifest, "arms": arms, "passed": passed, "reason": reason}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(("PASS: " if passed else "FAIL: ") + reason)
    print(f"artifacts: {out_dir}")

    if any(arm["trace_file"] is None for arm in arms.values()):
        print("ABORT: at least one arm produced no trace artifact.", file=sys.stderr)
        return 2
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
