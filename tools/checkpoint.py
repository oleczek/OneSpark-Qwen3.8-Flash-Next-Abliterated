#!/usr/bin/env python3
"""Offline checkpoint/artifact verification; Python standard library only.

No model downloads, imports of model code, or pickle deserialization here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def selected_snapshot(lock, env):
    if env.get("MODEL_SNAPSHOT"):
        return Path(env["MODEL_SNAPSHOT"]).expanduser().absolute()
    home = Path(env.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    hub = Path(env.get("HF_HUB_CACHE", str(home / "hub")))
    return hub / ("models--" + lock["model_id"].replace("/", "--")) / "snapshots" / lock["revision"]


def inspect_snapshot(snapshot, lock, full=False):
    """Check this one snapshot, never aggregate files across cached revisions."""
    snapshot = Path(snapshot)
    for name, expected in [("config.json", lock["config_sha256"]),
                           ("model.safetensors.index.json", lock["index_sha256"])]:
        if sha256(snapshot / name) != expected:
            raise ValueError(f"{name} differs from the pinned checkpoint")
    for name in ["tokenizer.json", "tokenizer_config.json"]:
        if not (snapshot / name).is_file():
            raise ValueError(f"Missing {name}; download the complete snapshot")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    if set(index["weight_map"].values()) != {f["name"] for f in lock["files"]}:
        raise ValueError("Index and pinned shard manifest disagree")
    records = []
    for spec in lock["files"]:
        name = spec["name"]
        if Path(name).name != name:
            raise ValueError("Only flat checkpoint shard names are supported")
        path = snapshot / name
        stat = path.stat()  # follows HF cache symlinks; broken links fail
        if stat.st_size != spec["size"]:
            raise ValueError(f"Wrong size or incomplete download: {name}")
        if full and sha256(path) != spec["sha256"]:
            raise ValueError(f"SHA256 mismatch (wrong model or damaged file): {name}")
        records.append([name, stat.st_size, stat.st_mtime_ns, stat.st_ino])
    # Include tokenizer contents: same-sized local edits must invalidate setup.
    tokenizers = {name: sha256(snapshot / name)
                  for name in ["tokenizer.json", "tokenizer_config.json"]}
    return {"snapshot": str(snapshot.resolve()), "revision": lock["revision"],
            "files": records, "tokenizers": tokenizers}


def artifact_metadata(root=ROOT):
    lock = json.loads((root / "model.lock.json").read_text())
    return {"format_version": 2, "model_id": lock["model_id"],
            "revision": lock["revision"], "R": 4,
            "source_manifest_sha256": sha256(root / "model.lock.json"),
            "builder_sha256": sha256(root / "tools/build_hashk_ple.py")}


def verify_artifact(path, expected, full=True):
    path = Path(path)
    metadata = json.loads(Path(str(path) + ".json").read_text())
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"HashK {key} mismatch; rebuild in this checkout")
    if metadata.get("bytes") != path.stat().st_size:
        raise ValueError("HashK size mismatch; incomplete artifact")
    if full and sha256(path) != metadata.get("sha256"):
        raise ValueError("HashK SHA256 mismatch; rebuild")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["snapshot", "artifact"])
    parser.add_argument("--full", action="store_true", help="Hash all 135 GB of checkpoint shards")
    args = parser.parse_args()
    if args.action == "artifact":
        verify_artifact(ROOT / "ple_hashk_R4.pt", artifact_metadata())
        print("HashK provenance and SHA256 OK")
        return
    lock = json.loads((ROOT / "model.lock.json").read_text())
    snapshot = selected_snapshot(lock, os.environ)
    result = inspect_snapshot(snapshot, lock, args.full)
    result["source_manifest_sha256"] = sha256(ROOT / "model.lock.json")
    receipt = ROOT / ".state/checkpoint.json"
    if args.full:
        atomic_json(receipt, result)
        print(f"Verified {len(result['files'])} shards against published SHA256s", file=sys.stderr)
    elif not receipt.is_file() or json.loads(receipt.read_text()) != result:
        raise ValueError("Checkpoint not verified here, or changed since verification; run ./install.sh")
    print(snapshot)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        sys.exit(f"Checkpoint check failed: {exc}")
