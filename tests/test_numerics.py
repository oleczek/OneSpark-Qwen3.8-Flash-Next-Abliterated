"""Small CPU tests only. Optional torch/safetensors deps; never allocate real PLE."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
try:
    import torch
    import sympy
    from safetensors.torch import save_file
    import build_hashk_ple as builder
except ImportError:
    torch = None


def extract_functions(source, names, scope):
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != set(names):
        raise AssertionError('Function anchors changed')
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<extracted CPU functions>', 'exec'), scope)
    return scope


@unittest.skipIf(torch is None, 'Optional CPU dependencies: torch, sympy, safetensors')
class NumericsTests(unittest.TestCase):
    def test_signed_hash_matches_scalar_reference_and_runtime(self):
        source = (ROOT / 'patches/qwen4_exp_nvfp4.py').read_text()
        runtime = extract_functions(source, ['_hashk_lsr', '_hashk_splitmix'], {})
        local = torch.tensor([0, 1, 20000000, 123456, 19999999], dtype=torch.int64)
        heads = torch.tensor([0, 15, 4, 9, 1], dtype=torch.int64)
        sizes = torch.tensor([5000001, 5000043, 5000015, 5000024, 5000006], dtype=torch.int64)
        mask = (1 << 64) - 1
        for sub in (0, 1):
            slots = builder.hash_slot(local, heads, sub, sizes)
            x = (local + 1) * 2862933555777941757 + builder.SALTS[sub] + heads * builder.HSALT
            self.assertTrue(torch.equal(slots, runtime['_hashk_splitmix'](x).remainder(sizes)))
            expected = []
            for value, head, size in zip(local.tolist(), heads.tolist(), sizes.tolist()):
                z = ((value + 1) * 2862933555777941757 + builder.SALTS[sub] + head * builder.HSALT + builder.GAMMA) & mask
                z = ((z ^ (z >> 30)) * builder.M1) & mask
                z = ((z ^ (z >> 27)) * builder.M2) & mask
                z ^= z >> 31
                if z >= 1 << 63:
                    z -= 1 << 64
                expected.append(z % size)
            self.assertEqual(slots.tolist(), expected)

    def test_all_head_boundaries_are_half_open(self):
        sizes, offsets, total = builder.head_sizes()
        boundary = torch.tensor(offsets[1:] + [total], dtype=torch.int64)
        ids = torch.tensor(offsets[1:], dtype=torch.int64)
        self.assertEqual(torch.bucketize(ids, boundary, right=True).tolist(), list(range(1, 16)))
        # Reproduces the original off-by-one: 15 exact boundaries went to prior head.
        self.assertEqual(torch.bucketize(ids, boundary, right=False).tolist(), list(range(15)))
        padded = ((total + 127) // 128) * 128
        self.assertEqual(padded, 320001536)

    def test_in_place_mean_preserves_values_and_storage(self):
        sums = torch.tensor([[2., 4.], [0., 0.], [-6., 9.]])
        counts = torch.tensor([2., 0., 3.])
        expected = sums / counts.clamp_min(1).unsqueeze(1)
        pointer = sums.data_ptr()
        actual = sums.div_(counts.clamp_min_(1).unsqueeze(1))
        self.assertEqual(actual.data_ptr(), pointer)
        self.assertTrue(torch.equal(actual, expected))

    def test_fp8_layout_scale_and_zero_rows(self):
        tree = ast.parse((ROOT / 'patch/patch_fp8.py').read_text())
        block = next(ast.literal_eval(node.value) for node in tree.body
                     if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'BLOCK' for t in node.targets))
        scope = extract_functions(block, ['_q4x_quantize_to_fp8_t'], {'torch': torch})
        torch.manual_seed(7)
        weights = torch.randn(17, 32).to(torch.bfloat16)
        weights[0].zero_()
        quantized, scales = scope['_q4x_quantize_to_fp8_t'](weights, chunk=5)
        self.assertEqual(tuple(quantized.shape), (32, 17))
        self.assertEqual(quantized.stride(), (1, 32))
        self.assertEqual(tuple(scales.shape), (17, 1))
        restored = quantized.t().float() * scales
        self.assertTrue(torch.isfinite(restored).all())
        self.assertTrue(torch.equal(restored[0], torch.zeros(32)))
        row_error = (restored - weights.float()).abs().amax(dim=1)
        self.assertTrue((row_error <= weights.float().abs().amax(dim=1) * 0.063 + 1e-6).all())

    def test_full_toy_builder_and_safe_artifact_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / 'toy.pt'
            # 2 heads x 4 dimensions; ~2k rows, not the 320M-row production table.
            overrides = {'HEADS': 2, 'DIM': 4, 'HALF': 2, 'SPLIT_PARTS': 2,
                         'NGRAM_BASE': 1000, 'DEV': 'cpu', 'OUT': str(out)}
            with patch.multiple(builder, **overrides):
                sizes, offsets, total = builder.head_sizes()
                rows = (total + 1) // 2
                weight_map = {}
                torch.manual_seed(2)
                for shard in range(2):
                    name = f'model.ple.ngram_embedding.shard_{shard}.weight'
                    filename = f'{shard}.safetensors'
                    save_file({name: (torch.randn(rows, 4) * .1).to(torch.float8_e4m3fn)}, str(root / filename))
                    weight_map[name] = filename
                (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weight_map}))
                with patch.object(builder, 'resolve_snapshot', return_value=str(root)), \
                     patch.object(torch.cuda, 'empty_cache'), contextlib.redirect_stdout(io.StringIO()):
                    builder.main()
                art = torch.load(out, map_location='cpu', weights_only=True)
                self.assertEqual(art['heads'], 2)
                self.assertEqual(art['A'].dtype, torch.float8_e4m3fn)
                self.assertTrue(torch.isfinite(art['W'].float()).all())
                import checkpoint
                checkpoint.verify_artifact(out, checkpoint.artifact_metadata())
                self.assertEqual(out.stat().st_mode & 0o777, 0o644)
                self.assertEqual(Path(str(out) + '.json').stat().st_mode & 0o777, 0o644)
                # Reuse is explicit: second build must not overwrite the first.
                before = checkpoint.sha256(out)
                with self.assertRaises(SystemExit):
                    builder.main()
                self.assertEqual(before, checkpoint.sha256(out))

    def test_no_snapshot_never_falls_back_or_downloads(self):
        with patch.object(builder, 'SNAPSHOT', ''):
            with self.assertRaises(SystemExit):
                builder.resolve_snapshot()


if __name__ == '__main__':
    unittest.main()
