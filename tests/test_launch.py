"""Shell integration against fake Docker/GPU/memory/HTTP tools; no real services."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'checkout with spaces'
        self.root.mkdir()
        for name in ['common.sh', 'launch.sh', 'stop.sh']:
            shutil.copy2(ROOT / name, self.root / name)
        self.bin = Path(self.tmp.name) / 'bin'
        self.bin.mkdir()
        self.trace = Path(self.tmp.name) / 'trace.jsonl'
        mock = '''#!PYTHON
import json,os,sys
from pathlib import Path
tool=Path(sys.argv[0]).name
args=sys.argv[1:]
with open(os.environ['TEST_TRACE'],'a') as stream:
    stream.write(json.dumps([tool]+args)+'\\n')
if tool=='uname': print('Linux')
elif tool=='awk': print('120')
elif tool=='python3':
    if args and args[0].endswith('checkpoint.py') and 'snapshot' in args: print('/fixture hub/snapshot')
    elif args and args[0].endswith('prepare_runtime.py'): print('sha256:'+'a'*64)
elif tool=='docker':
    if args[0]=='inspect':
        if '--format' not in args: sys.exit(0 if os.environ.get('TEST_EXISTS')=='1' else 1)
        if 'State.Running' in args[2]: print('true')
        else: print(os.environ.get('TEST_OWNER','unrelated checkout'))
    elif args[0]=='run':
        if os.environ.get('TEST_FAIL_RUN')=='1': sys.exit(125)
        print('fake-container-id')
'''.replace('PYTHON', sys.executable)
        for tool in ['docker', 'python3', 'uname', 'awk', 'curl', 'sleep']:
            path = self.bin / tool
            path.write_text(mock)
            path.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.bin) + ':' + os.environ['PATH'],
                        API_KEY='test-key-not-a-secret', TEST_TRACE=str(self.trace),
                        MODE='1', PORT='31337', THINKING='off')

    def run_script(self, name):
        return subprocess.run(['bash', str(self.root / name)], env=self.env, capture_output=True, text=True)

    def calls(self):
        if not self.trace.exists():
            return []
        return [json.loads(line) for line in self.trace.read_text().splitlines()]

    def test_launch_preserves_stack_and_authenticates_health(self):
        result = self.run_script('launch.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = self.calls()
        run = next(cmd for cmd in commands if cmd[:2] == ['docker', 'run'])
        self.assertNotIn('--privileged', run)
        for flag in ['--language-only', '--mamba-track-interval', '--speculative-num-draft-tokens', '--served-model-name']:
            self.assertIn(flag, run)
        self.assertEqual(run[run.index('--model-path') + 1], '/fixture hub/snapshot')
        self.assertEqual(run[run.index('--host') + 1], '127.0.0.1')
        self.assertEqual(run[run.index('--port') + 1], '31337')
        self.assertTrue(any('.state/qwen4_exp.py:' in part for part in run))
        curl = next(cmd for cmd in commands if cmd[0] == 'curl')
        self.assertIn('Authorization: Bearer test-key-not-a-secret', curl)
        self.assertNotIn(self.env['API_KEY'], result.stdout + result.stderr)
        self.assertFalse(any(cmd[:2] == ['docker', 'rm'] for cmd in commands))

    def test_failed_docker_run_stops_before_health_poll(self):
        self.env['TEST_FAIL_RUN'] = '1'
        self.assertNotEqual(self.run_script('launch.sh').returncode, 0)
        self.assertFalse(any(cmd[0] == 'curl' for cmd in self.calls()))

    def test_existing_container_is_never_replaced(self):
        self.env['TEST_EXISTS'] = '1'
        result = self.run_script('launch.sh')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(cmd[:2] in (['docker', 'run'], ['docker', 'rm'], ['docker', 'stop']) for cmd in self.calls()))

    def test_missing_key_fails_before_container_creation(self):
        self.env['API_KEY'] = ''
        self.assertNotEqual(self.run_script('launch.sh').returncode, 0)
        self.assertFalse(any(cmd[:2] == ['docker', 'run'] for cmd in self.calls()))

    def test_stop_refuses_unrelated_owner(self):
        self.env['TEST_EXISTS'] = '1'
        self.assertNotEqual(self.run_script('stop.sh').returncode, 0)
        self.assertFalse(any(cmd[:2] == ['docker', 'stop'] for cmd in self.calls()))

    def test_stop_removes_only_owned_container(self):
        self.env.update(TEST_EXISTS='1', TEST_OWNER=str(self.root))
        result = self.run_script('stop.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(['docker', 'stop', '--time', '60', 'onespark-abliterated'], self.calls())
        self.assertIn(['docker', 'rm', 'onespark-abliterated'], self.calls())

    def test_ple_off_does_not_require_hashk(self):
        self.env['MODE'] = '2'
        result = self.run_script('launch.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any('artifact' in cmd for cmd in self.calls()))
        run = next(cmd for cmd in self.calls() if cmd[:2] == ['docker', 'run'])
        self.assertIn('SGLANG_QWEN4_PLE_OFF=1', run)


if __name__ == '__main__':
    unittest.main()
