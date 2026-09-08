#!/usr/bin/env python3
"""Step 2: PLE_MODE=off -- drop the n-gram table entirely.

Anchored edit against flashnext-one-spark/patches/qwen4_exp_nvfp4.py, mirroring
the existing hashk path (which already deletes the embedding parameter, so the
mechanism is proven -- we only add "and load nothing in its place").

Three hooks:
  1. __init__      delete the weight parameter, flag the module
  2. gather        return zeros instead of looking anything up
  3. weight loader consume + discard the 10 model-plefp8-* shards, never allocate

The injection is `hidden_states = hidden_states + self.ple(...)`, so zeros make it
an exact no-op. RMSNorm(0) = 0/sqrt(eps) = 0, no NaN.

Frees 12.8 GB vs hashk (51.2 GB vs an unmodified checkpoint).

  --check   write to a temp copy and validate only; do NOT touch the live file
"""
import argparse, ast, os, pathlib, shutil, sys, tempfile

TARGET = pathlib.Path(os.environ["Q4X_TARGET"])

HELPER = '''

def _ple_off() -> bool:
    """PLE_MODE=off: no n-gram table at all. Not a speed lever -- the table is
    ~5 KB/token -- but it frees 12.8 GB and answers what the table is worth."""
    import os

    return os.environ.get("SGLANG_QWEN4_PLE_OFF", "0") == "1"

'''
HELPER_ANCHOR = "\ndef _nvfp4_ple_enabled() -> bool:"

INIT_ANCHOR = """        if _hashk_path():
            if getattr(config, "ple_offload_embedding", False):
                config.ple_offload_embedding = False"""
INIT_NEW = """        if _ple_off():
            if getattr(config, "ple_offload_embedding", False):
                config.ple_offload_embedding = False
            del self.ngram_embedding._parameters["weight"]
            torch.cuda.empty_cache()
            self.ngram_embedding.ple_off = True
            logger.info(
                "PLE OFF: n-gram embedding deleted, injection zeroed "
                "(frees 12.8 GB vs hashk, 51.2 GB vs the raw checkpoint)"
            )
        elif _hashk_path():
            if getattr(config, "ple_offload_embedding", False):
                config.ple_offload_embedding = False"""

GATHER_ANCHOR = """        if getattr(self.ngram_embedding, "hashk_mode", False):
            embeddings = _hashk_gather(self.ngram_embedding, lookup_ids)"""
GATHER_NEW = """        if getattr(self.ngram_embedding, "ple_off", False):
            # [T, heads] -> [T, heads, head_dim] of zeros. The PLE output is added
            # to the residual stream, so zeros make the whole injection a no-op.
            embeddings = torch.zeros(
                (*lookup_ids.shape, self.head_dim_per_ngram),
                dtype=torch.bfloat16,
                device=lookup_ids.device,
            )
        elif getattr(self.ngram_embedding, "hashk_mode", False):
            embeddings = _hashk_gather(self.ngram_embedding, lookup_ids)"""

LOADER_ANCHOR = """            if getattr(emb, "hashk_mode", False):
                loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
                return True"""
LOADER_NEW = """            if getattr(emb, "ple_off", False) or getattr(emb, "hashk_mode", False):
                # Consume and discard: the 10 model-plefp8-* shards are read off
                # disk and dropped, never allocated.
                loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
                return True"""

EDITS = [("helper", HELPER_ANCHOR, HELPER + HELPER_ANCHOR),
         ("init", INIT_ANCHOR, INIT_NEW),
         ("gather", GATHER_ANCHOR, GATHER_NEW),
         ("loader", LOADER_ANCHOR, LOADER_NEW)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="validate against a temp copy; leave the live file alone")
    args = ap.parse_args()

    src = TARGET.read_text()
    if "_ple_off" in src:
        if any(src.count(repl) != 1 for _, _, repl in EDITS):
            raise ValueError("Partial or changed PLE-off patch; restore pristine source")
        ast.parse(src)
        print("already patched; nothing to do")
        return 0

    out = src
    for name, anchor, repl in EDITS:
        if out.count(anchor) != 1:
            print(f"FATAL: anchor {name!r} not found -- refusing to guess", file=sys.stderr)
            return 1
        out = out.replace(anchor, repl, 1)
        print(f"  applied: {name}")

    ast.parse(out)
    print("syntax OK")

    if args.check:
        print(f"--check: validated in memory (+{len(out) - len(src)} bytes); target untouched")
        return 0

    backup = TARGET.with_suffix(".py.prestep2")
    if not backup.exists():
        shutil.copy2(TARGET, backup)
        print(f"backup -> {backup.name}")
    fd, tmp = tempfile.mkstemp(prefix=TARGET.name + ".", dir=TARGET.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(out)
        os.replace(tmp, TARGET)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f"patched {TARGET.name}: +{len(out) - len(src)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
