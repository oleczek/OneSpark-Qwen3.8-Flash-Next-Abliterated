import ast
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import checkpoint
import prepare_runtime
import api_probe


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = self.root / "snapshots" / "pinned"
        self.snapshot.mkdir(parents=True)
        for name, content in {
            "config.json": '{"arch":"fixture"}', "tokenizer.json": '{}',
            "tokenizer_config.json": '{}', "a.safetensors": 'fixture-weights',
            "model.safetensors.index.json": '{"weight_map":{"weight":"a.safetensors"}}',
        }.items():
            (self.snapshot / name).write_text(content)
        self.lock = {"model_id": "owner/model", "revision": "pinned",
                     "config_sha256": checkpoint.sha256(self.snapshot / "config.json"),
                     "index_sha256": checkpoint.sha256(self.snapshot / "model.safetensors.index.json"),
                     "files": [{"name": "a.safetensors", "size": 15,
                                "sha256": checkpoint.sha256(self.snapshot / "a.safetensors")}]}

    def test_full_verification(self):
        result = checkpoint.inspect_snapshot(self.snapshot, self.lock, full=True)
        self.assertEqual(len(result["files"]), 1)

    def test_same_size_corruption_is_rejected(self):
        (self.snapshot / "a.safetensors").write_text('wrong--weights!')
        # Keep the exact expected size: a size-only check cannot prove identity.
        (self.snapshot / "a.safetensors").write_bytes(b'x' * 15)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            checkpoint.inspect_snapshot(self.snapshot, self.lock, full=True)

    def test_other_snapshot_cannot_fill_missing_shard(self):
        other = self.root / "snapshots/other"
        other.mkdir()
        (self.snapshot / "a.safetensors").rename(other / "a.safetensors")
        with self.assertRaises(FileNotFoundError):
            checkpoint.inspect_snapshot(self.snapshot, self.lock)

    def test_broken_cache_symlink(self):
        path = self.snapshot / "a.safetensors"
        path.unlink()
        path.symlink_to("../../blobs/missing")
        with self.assertRaises(FileNotFoundError):
            checkpoint.inspect_snapshot(self.snapshot, self.lock)

    def test_config_mismatch(self):
        (self.snapshot / "config.json").write_text('{}')
        with self.assertRaisesRegex(ValueError, "config.json differs"):
            checkpoint.inspect_snapshot(self.snapshot, self.lock)

    def test_exact_revision_and_custom_hub(self):
        path = checkpoint.selected_snapshot(self.lock, {"HF_HUB_CACHE": "/custom hub"})
        self.assertEqual(str(path), "/custom hub/models--owner--model/snapshots/pinned")
        explicit = checkpoint.selected_snapshot(self.lock, {"MODEL_SNAPSHOT": "/local/model"})
        self.assertEqual(str(explicit), "/local/model")

    def test_artifact_binding_and_checksum(self):
        path = self.root / "hashk.pt"
        path.write_bytes(b'tensor-fixture')
        expected = {"model_id": "owner/model", "revision": "pinned", "R": 4}
        metadata = dict(expected, bytes=path.stat().st_size, sha256=checkpoint.sha256(path))
        checkpoint.atomic_json(str(path) + ".json", metadata)
        checkpoint.verify_artifact(path, expected)
        with self.assertRaisesRegex(ValueError, "revision mismatch"):
            checkpoint.verify_artifact(path, dict(expected, revision="another"))
        path.write_bytes(b'x' * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            checkpoint.verify_artifact(path, expected)

    def test_legacy_artifact_is_not_silently_reused(self):
        path = self.root / "legacy.pt"
        path.write_bytes(b'legacy')
        with self.assertRaises(FileNotFoundError):
            checkpoint.verify_artifact(path, {})


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "checkout with spaces"
        shutil.copytree(ROOT, self.root, ignore=shutil.ignore_patterns('.state', '__pycache__', '*.pt', 'results', '.git'))
        self.target = Path(self.tmp.name) / "model.py"
        shutil.copy2(self.root / "patches/qwen4_exp_nvfp4.py", self.target)

    def run_patch(self, name):
        return subprocess.run([sys.executable, str(self.root / "patch" / name)],
                              env=dict(os.environ, Q4X_TARGET=str(self.target)),
                              capture_output=True, text=True)

    def test_patches_are_idempotent(self):
        for name in ['patch_fp8.py', 'patch_ple_off.py']:
            self.assertEqual(self.run_patch(name).returncode, 0)
        expected = self.target.read_bytes()
        for name in ['patch_fp8.py', 'patch_ple_off.py']:
            result = self.run_patch(name)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(expected, self.target.read_bytes())

    def test_partial_patch_fails_closed(self):
        self.target.write_text(self.target.read_text() + '\n# Q4X step 1\n')
        before = self.target.read_bytes()
        self.assertNotEqual(self.run_patch('patch_fp8.py').returncode, 0)
        self.assertEqual(before, self.target.read_bytes())

    def test_ambiguous_anchor_fails_closed(self):
        self.target.write_text(self.target.read_text() + '\nEntryClass = [Qwen4ExpForConditionalGeneration]\n')
        before = self.target.read_bytes()
        self.assertNotEqual(self.run_patch('patch_fp8.py').returncode, 0)
        self.assertEqual(before, self.target.read_bytes())

    def test_bad_syntax_does_not_overwrite_source(self):
        self.target.write_text(self.target.read_text() + '\ninvalid syntax (\n')
        before = self.target.read_bytes()
        self.assertNotEqual(self.run_patch('patch_fp8.py').returncode, 0)
        self.assertEqual(before, self.target.read_bytes())

    def test_generated_overlay_and_image_lock(self):
        before = (self.root / "patches/qwen4_exp_nvfp4.py").read_bytes()
        image_id = "sha256:" + 'a' * 64
        prepare_runtime.prepare(self.root, image_id, "image:test")
        self.assertEqual(prepare_runtime.check(self.root), image_id)
        overlay = (self.root / ".state/qwen4_exp.py").read_text()
        self.assertIn('weights_only=True', overlay)
        self.assertEqual(before, (self.root / "patches/qwen4_exp_nvfp4.py").read_bytes())
        (self.root / ".state/qwen4_exp.py").write_text(overlay + '\n# changed\n')
        with self.assertRaisesRegex(ValueError, "overlay changed"):
            prepare_runtime.check(self.root)

    def test_all_python_parses_without_importing_kernels(self):
        for file in self.root.rglob('*.py'):
            with self.subTest(file=file.name):
                ast.parse(file.read_text())


class APITests(unittest.TestCase):
    def result(self, content=None, **kwargs):
        return {"choices": [{"message": dict(content=content, **kwargs)}],
                "usage": {"completion_tokens": 5}}

    def test_auth_custom_url_and_json_body(self):
        client = api_probe.Client('http://127.0.0.1:12345', 'test-only-key')
        with patch.object(client.opener, 'open', return_value=io.BytesIO(json.dumps(self.result('4')).encode())) as opened:
            result = client.chat('model', 'What is 2 + 2?')
        request = opened.call_args[0][0]
        self.assertEqual(request.full_url, 'http://127.0.0.1:12345/v1/chat/completions')
        self.assertEqual(request.get_header('Authorization'), 'Bearer test-only-key')
        self.assertFalse(json.loads(request.data)['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(result['completion_tokens'], 5)

    def test_http_error_is_not_a_benchmark_success(self):
        client = api_probe.Client('http://localhost', 'test-only-key')
        with patch.object(client.opener, 'open', side_effect=HTTPError('url', 401, 'unauthorized', {}, None)):
            with self.assertRaises(HTTPError):
                client.chat('model', 'hello')

    def test_empty_response_fails(self):
        client = api_probe.Client('http://localhost', 'test-only-key')
        with patch.object(client.opener, 'open', return_value=io.BytesIO(json.dumps(self.result('')).encode())):
            with self.assertRaisesRegex(ValueError, 'Empty'):
                client.chat('model', 'hello')

    def test_semantic_smoke_validators(self):
        api_probe.validate_math({'response': self.result('4')})
        api_probe.validate_json({'response': self.result('{"ok":true,"value":7}')})
        api_probe.validate_reasoning({'response': self.result('323', reasoning_content='17 * (20 - 1) = 340 - 17')})
        api_probe.validate_tool({'response': self.result(None, tool_calls=[{
            'function': {'name': 'get_weather', 'arguments': '{"city":"Warsaw"}'}}])})
        with self.assertRaises(ValueError):
            api_probe.validate_math({'response': self.result('!!!!')})

    def test_redirect_does_not_forward_key(self):
        self.assertIsNone(api_probe.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://elsewhere'))

    def test_prompt_budgets_are_shared_by_serial_and_parallel(self):
        pools = json.loads((ROOT / 'tools/bench_prompts.json').read_text())
        self.assertEqual([row['max_tokens'] for row in pools], [256, 512, 1024, 64, 2048])
        self.assertTrue(all(len(row['prompts']) == 8 for row in pools))

    def test_parallel_rate_uses_one_wall_clock_and_propagates_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'tools').mkdir()
            (root / 'tools/bench_prompts.json').write_text(json.dumps([
                {'name': 'fixture', 'max_tokens': 77, 'prompts': ['one', 'two']}]))
            client = Mock()
            client.chat.side_effect = [{'completion_tokens': 10}, {'completion_tokens': 20}]
            with patch.object(api_probe, 'ROOT', root), patch.object(api_probe.time, 'perf_counter', side_effect=[0, 10]):
                batch = api_probe.benchmark(client, 'model', 2, 1)[0]
            self.assertEqual(batch['aggregate_e2e_output_tokens_per_second'], 3)
            self.assertTrue(batch['ok'])
            self.assertTrue(all(call.args[2] == 77 for call in client.chat.call_args_list))
            client.chat.side_effect = [{'completion_tokens': 10}, ValueError('failed stream')]
            with patch.object(api_probe, 'ROOT', root), patch.object(api_probe.time, 'perf_counter', side_effect=[0, 10]):
                self.assertFalse(api_probe.benchmark(client, 'model', 2, 1)[0]['ok'])

    def test_long_canary_validates_content_and_actual_token_count(self):
        client = Mock()
        def request(path, payload=None, authenticated=True):
            if not authenticated:
                raise HTTPError('url', 401, 'unauthorized', {}, None)
        client.request.side_effect = request
        def chat(model, prompt, **kwargs):
            if prompt.startswith('Reference entry'):
                import re
                code = re.search(r'The unique retrieval code is ([a-f0-9]+)', prompt)[1]
                response = self.result(code)
                response['usage']['prompt_tokens'] = 10000
                return {'response': response}
            if '2 + 2' in prompt:
                return {'response': self.result('4')}
            if prompt.startswith('Return only this JSON'):
                return {'response': self.result('{"ok":true,"value":7}')}
            if 'get_weather' in prompt:
                return {'response': self.result(None, tool_calls=[{'function': {'name': 'get_weather', 'arguments': '{"city":"Warsaw"}'}}])}
            return {'response': self.result('323', reasoning_content='verified')}
        client.chat.side_effect = chat
        checks = api_probe.smoke(client, 'model', long_probe=True)
        self.assertEqual(len(checks), 6)
        self.assertTrue(all(check['ok'] for check in checks))


if __name__ == '__main__':
    unittest.main()
