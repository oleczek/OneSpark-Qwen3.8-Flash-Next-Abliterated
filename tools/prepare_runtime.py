#!/usr/bin/env python3
"""Generate overlays from unmodified vendored sources; lock the local image ID."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from checkpoint import ROOT, atomic_json, sha256


def source_hashes(root):
    paths = [root / "model.lock.json"]
    paths += sorted((root / "patches").rglob("*.py"))
    paths += sorted((root / "patch").glob("*.py"))
    paths += [root / "tools" / name for name in
              ["prepare_runtime.py", "checkpoint.py", "build_hashk_ple.py"]]
    return {str(path.relative_to(root)): sha256(path) for path in paths}


def prepare(root, image_id, image_ref):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("Expected immutable local Docker image ID")
    state = root / ".state"
    state.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="overlay-", dir=state) as tmp:
        target = Path(tmp) / "qwen4_exp.py"
        shutil.copy2(root / "patches/qwen4_exp_nvfp4.py", target)
        env = dict(os.environ, Q4X_TARGET=str(target))
        for patcher in ["patch_fp8.py", "patch_ple_off.py"]:
            subprocess.run([sys.executable, str(root / "patch" / patcher)],
                           env=env, check=True)
        source = target.read_text()
        anchor = 'torch.load(_hashk_path(), map_location="cpu", weights_only=False)'
        if source.count(anchor) != 1:
            raise ValueError("Unexpected HashK loader; refusing to guess")
        source = source.replace(anchor, anchor.replace("False", "True"))
        ast.parse(source)
        target.write_text(source)
        os.replace(target, state / "qwen4_exp.py")
    atomic_json(state / "runtime.json", {
        "image_id": image_id, "image_ref": image_ref,
        "inputs": source_hashes(root), "overlay_sha256": sha256(state / "qwen4_exp.py"),
    })


def check(root):
    state = root / ".state"
    lock = json.loads((state / "runtime.json").read_text())
    if lock["inputs"] != source_hashes(root):
        raise ValueError("Runtime inputs changed; rerun ./install.sh")
    if lock["overlay_sha256"] != sha256(state / "qwen4_exp.py"):
        raise ValueError("Generated overlay changed; rerun ./install.sh")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", lock["image_id"]):
        raise ValueError("Invalid image ID in runtime receipt")
    return lock["image_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--image-id")
    parser.add_argument("--image-ref")
    parser.add_argument("--record-launch", nargs=7, metavar=('MODE', 'CTX', 'MEM', 'THINKING', 'PORT', 'HOST', 'CONTAINER'))
    args = parser.parse_args()
    if args.check:
        print(check(ROOT))
    elif args.record_launch:
        values = dict(zip(['mode', 'context_length', 'mem_fraction', 'thinking', 'port', 'listen_host', 'container'], args.record_launch))
        values['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
        values['image_id'] = check(ROOT)
        atomic_json(ROOT / '.state/last_launch.json', values)
    else:
        if not args.image_id or not args.image_ref:
            parser.error("--image-id and --image-ref are required")
        prepare(ROOT, args.image_id, args.image_ref)
        print("Generated overlay and locked local image ID; vendored sources untouched")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Runtime setup failed: {exc}")
