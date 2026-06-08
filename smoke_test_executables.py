"""
Lightweight smoke tests for the project scripts.

This script checks that the main executable Python files are present, parse as
valid Python, and that the expected data/checkpoint paths exist. It does not
train models or run full evaluations.

Optional:
  python smoke_test_executables.py --run-help

The --run-help mode only calls scripts with --help, which should exit before
loading datasets, checkpoints, or starting training.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


EXECUTABLES = [
    "mainpro_FER.py",
    "preprocess_fer2013.py",
    "plot_results.py",
    "plot_fer2013_confusion_matrix.py",
    "gradcamVisualize.py",
    "gradcam_comparison.py",
    "test_private_bestmodel_fer_test.py",
    "fine_tuning/01_inspect_rafce_mapping.py",
    "fine_tuning/02_zero_shot_private_model.py",
    "fine_tuning/03_fine_tune_private_model.py",
    "fine_tuning/04_evaluate_finetuned_model.py",
    "fine_tuning/05_test_best_model_on_fer.py",
    "occlusion_evaluation/evaluate_fer_test_center_black_occlusion.py",
]


HELP_SAFE_EXECUTABLES = [
    "mainpro_FER.py",
    "plot_fer2013_confusion_matrix.py",
    "gradcam_comparison.py",
    "test_private_bestmodel_fer_test.py",
    "fine_tuning/01_inspect_rafce_mapping.py",
    "fine_tuning/02_zero_shot_private_model.py",
    "fine_tuning/03_fine_tune_private_model.py",
    "fine_tuning/04_evaluate_finetuned_model.py",
    "fine_tuning/05_test_best_model_on_fer.py",
    "occlusion_evaluation/evaluate_fer_test_center_black_occlusion.py",
]


REQUIRED_PATHS = [
    "datasets/fer2013.csv",
    "datasets/data.h5",
    "checkpoints/PrivateTest_model.t7",
    "checkpoints/Best_model.t7",
]


OPTIONAL_DATASET_PATHS = [
    "FerDataset/test",
    "FerDataset/train",
    "RafceDataset/test/img",
    "RafceDataset/test/pre-processing/RAFCE_emolabel.txt",
    "RafceDataset/train/augmented_img",
    "RafceDataset/train/after-processing/RAFCE_emolabel.txt",
    "RafceDataset/validation/img",
    "RafceDataset/validation/pre-processing/RAFCE_emolabel.txt",
]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def check_file_exists(relative_path: str) -> CheckResult:
    path = PROJECT_ROOT / relative_path
    if path.exists():
        return CheckResult(relative_path, True)
    return CheckResult(relative_path, False, "missing")


def check_python_syntax(relative_path: str) -> CheckResult:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return CheckResult(relative_path, False, "missing")

    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except UnicodeDecodeError:
        return CheckResult(relative_path, False, "could not read as UTF-8")
    except SyntaxError as exc:
        return CheckResult(relative_path, False, f"syntax error at line {exc.lineno}: {exc.msg}")

    return CheckResult(relative_path, True)


def check_path(relative_path: str, required: bool) -> CheckResult:
    path = PROJECT_ROOT / relative_path
    if path.exists():
        if path.is_file():
            size = path.stat().st_size
            return CheckResult(relative_path, True, f"{size} bytes")
        return CheckResult(relative_path, True, "directory")

    label = "missing required path" if required else "missing optional path"
    return CheckResult(relative_path, not required, label)


def run_help(relative_path: str, timeout_seconds: int) -> CheckResult:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return CheckResult(relative_path, False, "missing")

    command = [sys.executable, str(path), "--help"]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CheckResult(relative_path, False, f"--help timed out after {timeout_seconds}s")

    if completed.returncode == 0:
        return CheckResult(relative_path, True)

    stderr = completed.stderr.strip().splitlines()
    stdout = completed.stdout.strip().splitlines()
    message = stderr[-1] if stderr else (stdout[-1] if stdout else f"exit code {completed.returncode}")
    return CheckResult(relative_path, False, message)


def print_section(title: str, results: list[CheckResult]) -> int:
    print(f"\n{title}")
    print("-" * len(title))

    failures = 0
    for result in results:
        marker = "OK" if result.ok else "FAIL"
        detail = f" ({result.detail})" if result.detail else ""
        print(f"[{marker}] {result.name}{detail}")
        if not result.ok:
            failures += 1

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight project smoke tests.")
    parser.add_argument(
        "--run-help",
        action="store_true",
        help="also execute --help for scripts where that should not start training or evaluation",
    )
    parser.add_argument(
        "--strict-optional-data",
        action="store_true",
        help="fail if optional external dataset folders are missing",
    )
    parser.add_argument(
        "--timeout",
        default=15,
        type=int,
        help="timeout in seconds for each --help command",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures = 0

    failures += print_section(
        "Executable files",
        [check_file_exists(path) for path in EXECUTABLES],
    )

    failures += print_section(
        "Python syntax",
        [check_python_syntax(path) for path in EXECUTABLES],
    )

    failures += print_section(
        "Required project paths",
        [check_path(path, required=True) for path in REQUIRED_PATHS],
    )

    optional_results = [check_path(path, required=False) for path in OPTIONAL_DATASET_PATHS]
    if args.strict_optional_data:
        optional_results = [
            CheckResult(result.name, result.ok and "missing optional path" not in result.detail, result.detail)
            for result in optional_results
        ]
    failures += print_section("Optional dataset paths", optional_results)

    if args.run_help:
        failures += print_section(
            "CLI --help checks",
            [run_help(path, args.timeout) for path in HELP_SAFE_EXECUTABLES],
        )
    else:
        print("\nCLI --help checks")
        print("-----------------")
        print("Skipped. Run with --run-help to check argparse entry points without training.")

    if failures:
        print(f"\nSmoke test finished with {failures} failing check(s).")
        return 1

    print("\nSmoke test finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
