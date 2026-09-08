#!/usr/bin/env python3
"""Step 1: post-load FP8 cast for lm_head (and, later, other dense modules).

Anchored edit against flashnext-one-spark/patches/qwen4_exp_nvfp4.py, following
that repo's generator convention. Idempotent; keeps a .orig backup.

Enable at runtime with  Q4X_FP8=lm_head  in the container env. Unset = no-op,
so the patched file is safe to leave bind-mounted for a baseline re-run.

Why lm_head: NEXTN reads it 4x per iteration (3 draft steps + verify).
248320 x 2560 = 0.636B params, 1.27 GB BF16 -> 0.64 GB FP8, saving 2.5 GB/iteration.
"""
import ast, os, pathlib, shutil, sys, tempfile

TARGET = pathlib.Path(os.environ["Q4X_TARGET"])

BLOCK = '''

# ==== Q4X step 1: post-load FP8 cast of dense weights (env Q4X_FP8) ====
# Weight-only FP8 e4m3, per-output-channel absmax. No calibration data needed.
# Layout note: sglang's apply_fp8_linear wants weight as (K, N) with
# weight_scale.numel() == weight.shape[1]; nn.Linear stores (N, K). We transpose.
# Getting that backwards produces silent garbage, not an error.


class Q4XFp8LinearMethod:
    """Minimal weight-only FP8 apply(). The class NAME matters: logits_processor
    skips quant methods listed in _UNQUANTIZED_LM_HEAD_METHODS, and this is not
    one of them, so lm_head routes through apply() instead of the bf16 matmul."""

    def __init__(self, cutlass_ok: bool):
        self.cutlass_ok = cutlass_ok

    def process_weights_after_loading(self, layer):
        # Required: sglang's loader calls this on every module carrying a
        # quant_method. Our weights are already final -- cast in load_weights.
        return

    def create_weights(self, *args, **kwargs):  # never used; we attach post-load
        raise NotImplementedError("Q4XFp8LinearMethod is applied after loading")

    def apply(self, layer, x, bias=None):
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=None,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_ok,
            use_per_token_if_dynamic=True,
        )


def _q4x_fp8_targets():
    import os

    return {t.strip() for t in os.environ.get("Q4X_FP8", "").split(",") if t.strip()}


def _q4x_quantize_to_fp8_t(w: torch.Tensor, chunk: int = 8192):
    """(N, K) bf16 -> ((K, N) COLUMN-MAJOR fp8, (N, 1) fp32 scale).

    Layout is load-bearing: the cutlass fp8 GEMM rejects a row-major B with
    "mat_b must be a column major tensor". Build a contiguous (N, K) buffer and
    return its .t() view -- strides (1, K), i.e. column major -- which is exactly
    what sglang's own Fp8LinearMethod does (`Parameter(weight.t())`).
    Chunked over N so we never hold a second bf16 copy.
    """
    N, K = w.shape
    buf = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=w.device)
    scales = torch.empty((N, 1), dtype=torch.float32, device=w.device)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        blk = w[i:j].to(torch.float32)
        s = blk.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
        buf[i:j] = (blk / s).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
        scales[i:j] = s
        del blk, s
    out = buf.t()                      # (K, N) column-major view, no copy
    assert out.stride() == (1, K), f"expected column-major B, got {out.stride()}"
    return out, scales


def _q4x_apply_fp8(model):
    targets = _q4x_fp8_targets()
    if not targets:
        return
    from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported

    ok = cutlass_fp8_supported()

    if "lm_head" in targets:
        lm = getattr(model, "lm_head", None)
        if lm is None or not hasattr(lm, "weight"):
            logger.warning("Q4X_FP8: lm_head requested but not found -- SKIPPED")
        elif lm.weight.dtype == torch.float8_e4m3fn:
            logger.info("Q4X_FP8: lm_head already fp8, nothing to do")
        else:
            before = lm.weight.numel() * lm.weight.element_size()
            q, s = _q4x_quantize_to_fp8_t(lm.weight.data)
            lm.weight = torch.nn.Parameter(q, requires_grad=False)
            lm.weight_scale = torch.nn.Parameter(s, requires_grad=False)
            lm.input_scale = None
            lm.quant_method = Q4XFp8LinearMethod(ok)
            torch.cuda.empty_cache()
            after = q.numel() * q.element_size()
            # LOUD, because a silent no-op here is the whole failure mode: the
            # server would serve fine at baseline speed and we would misread it
            # as "fp8 did not help".
            logger.info(
                "Q4X_FP8 ACTIVE: lm_head %s -> fp8 %s  (%.2f -> %.2f GB, cutlass=%s)",
                tuple(lm.weight_scale.shape[:1]) + (q.shape[0],),
                tuple(q.shape), before / 1e9, after / 1e9, ok,
            )

    # --- hyper-connections: plain nn.Linear, no quant_method to swap ---
    # The compute takes raw weight TENSORS, so we attach packed fp8 tuples to the
    # GatedResidual module and let the patched hyperconnection.py pick them up.
    # ⚠ This DISABLES the fused Triton hc_mix kernel, whose guard requires
    # w.dtype == activation dtype. Expected to be a regression; measured on purpose.
    if "hyper_connection" in targets:
        n_hc = 0
        for name, mod in model.named_modules():
            if not hasattr(mod, "input_mix_weight_down"):
                continue
            try:
                mod._q4x_down = _q4x_quantize_to_fp8_t(mod.input_mix_weight_down.weight.data)
                mod._q4x_up = _q4x_quantize_to_fp8_t(mod.input_mix_weight_up.weight.data)
                if getattr(mod, "block_inject_weight", None) is not None:
                    mod._q4x_inject = _q4x_quantize_to_fp8_t(mod.block_inject_weight.weight.data)
                n_hc += 1
            except Exception as e:
                logger.warning("Q4X_FP8: hyper_connection %s FAILED (%s)", name, e)
        torch.cuda.empty_cache()
        if n_hc:
            logger.info(
                "Q4X_FP8 ACTIVE: %d hyper-connection blocks -> fp8 "
                "(fused Triton hc_mix is now BYPASSED for these)", n_hc,
            )
        else:
            logger.warning("Q4X_FP8: target 'hyper_connection' matched NO modules")

    # --- generic dense sweep: any sglang Linear whose qualified name matches ---
    # Selected by NAME MATCH over named_modules() rather than a hardcoded list,
    # and every conversion is logged, so a target that silently matches nothing
    # is visible instead of being scored as "fp8 did not help".
    module_targets = targets - {"lm_head", "hyper_connection"}
    if module_targets:
        converted, skipped, total_before, total_after = [], [], 0, 0
        for name, mod in model.named_modules():
            if not any(t in name for t in module_targets):
                continue
            w = getattr(mod, "weight", None)
            if w is None or w.dim() != 2 or w.dtype != torch.bfloat16:
                continue                      # norms (1-D), conv1d (3-D), already-quantized
            qm = getattr(mod, "quant_method", None)
            if qm is None:
                # plain nn.Linear (e.g. the hyper-connections): no quant_method to
                # swap, forward goes straight to F.linear. Needs a different
                # mechanism -- report it rather than pretending it was done.
                skipped.append((name, "no quant_method (plain nn.Linear)"))
                continue
            if type(qm).__name__ not in ("UnquantizedLinearMethod", "UnquantizedEmbeddingMethod"):
                skipped.append((name, f"already {type(qm).__name__}"))
                continue
            before = w.numel() * w.element_size()
            q, s = _q4x_quantize_to_fp8_t(w.data)
            mod.weight = torch.nn.Parameter(q, requires_grad=False)
            mod.weight_scale = torch.nn.Parameter(s, requires_grad=False)
            mod.input_scale = None
            mod.quant_method = Q4XFp8LinearMethod(ok)
            total_before += before
            total_after += q.numel() * q.element_size()
            converted.append(name)
            # Free eagerly and compact periodically: 200+ alloc/free pairs
            # otherwise fragment the caching allocator and SHRINK the KV pool,
            # which is sized from free memory after loading.
            del w
            if len(converted) % 16 == 0:
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        logger.info(
            "Q4X_FP8 ACTIVE: %d modules -> fp8 (%.2f -> %.2f GB, saved %.2f GB) for targets %s",
            len(converted), total_before / 1e9, total_after / 1e9,
            (total_before - total_after) / 1e9, sorted(module_targets),
        )
        for t_ in sorted(module_targets):
            n = sum(1 for c in converted if t_ in c)
            if n == 0:
                logger.warning("Q4X_FP8: target %r matched NO modules -- check the name", t_)
            else:
                logger.info("Q4X_FP8:   %-20s %4d modules", t_, n)
        for name, why in skipped[:12]:
            logger.warning("Q4X_FP8: SKIPPED %s (%s)", name, why)
        if len(skipped) > 12:
            logger.warning("Q4X_FP8: ... and %d more skipped", len(skipped) - 12)


# ==== end Q4X step 1 ====
'''

ANCHOR_HOOK = """        for module in self.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.finalize_fused_in_proj()

        return loaded_params"""

REPLACE_HOOK = """        for module in self.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.finalize_fused_in_proj()

        _q4x_apply_fp8(self)

        return loaded_params"""

ANCHOR_BLOCK = "\nEntryClass = [Qwen4ExpForConditionalGeneration]"


def main():
    src = TARGET.read_text()
    if "Q4X step 1" in src:
        if src.count(REPLACE_HOOK) != 1 or src.count(BLOCK) != 1:
            raise ValueError("Partial or changed FP8 patch; restore pristine source")
        ast.parse(src)
        print("already patched; nothing to do")
        return 0
    if src.count(ANCHOR_HOOK) != 1:
        print("FATAL: hook anchor not found -- refusing to guess", file=sys.stderr)
        return 1
    if src.count(ANCHOR_BLOCK) != 1:
        print("FATAL: block anchor not found -- refusing to guess", file=sys.stderr)
        return 1

    backup = TARGET.with_suffix(".py.orig")
    if not backup.exists():
        shutil.copy2(TARGET, backup)
        print(f"backup -> {backup.name}")

    out = src.replace(ANCHOR_HOOK, REPLACE_HOOK, 1)
    assert out != src
    out = out.replace(ANCHOR_BLOCK, BLOCK + ANCHOR_BLOCK, 1)
    ast.parse(out)          # validate BEFORE replacing the target
    fd, tmp = tempfile.mkstemp(prefix=TARGET.name + ".", dir=TARGET.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(out)
        os.replace(tmp, TARGET)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f"patched {TARGET.name}: +{len(out) - len(src)} bytes, syntax OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
