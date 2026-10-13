"""Run a command and retain actual stdout, stderr, exit status and wall time."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    root = Path(__file__).resolve().parents[1]
    paths = [*root.glob("src/htis_context/*.py"), *root.glob("tests/*.py"),
             *root.glob("configs/*.json"), *root.glob("scripts/*.py"), root / "pyproject.toml"]
    source_hashes = {str(p.relative_to(root)).replace("\\", "/"):
                     hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    start = time.perf_counter()
    stamp = datetime.now(timezone.utc).isoformat()
    try:
        run = subprocess.run(command, cwd=args.cwd, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
    except OSError as error:
        run = subprocess.CompletedProcess(command, 127, "", f"Launch failed: {type(error).__name__}: {error}\n")
    record = {"command": command, "cwd": str((args.cwd or Path.cwd()).resolve()),
              "started_at_utc": stamp, "wall_seconds": time.perf_counter() - start,
              "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    record["source_files_sha256"] = source_hashes
    record["source_modified_during_run"] = any(hashlib.sha256(p.read_bytes()).hexdigest() !=
        source_hashes[str(p.relative_to(root)).replace("\\", "/")] for p in paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(run.stdout, end="")
    print(run.stderr, end="", file=sys.stderr)
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
