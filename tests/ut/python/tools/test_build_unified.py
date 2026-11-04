"""Check native command routing and stop-before-merge behavior without compilation."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
WHEEL_URL = ('https://downloads.example.invalid/torch_npu-2.13.0%2Btest-cp310-cp310-'
             'manylinux_2_28_aarch64.whl?download=1&source=official')


class NativeBuildFlowTest(unittest.TestCase):
    """Exercise command routing with native build stubs and real Python interpreters."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        for name in ('scripts/packaging', 'third_party/torch_npu/ci', 'bin'):
            (self.root / name).mkdir(parents=True)
        shutil.copy2(ROOT / 'scripts/build_unified.sh', self.root / 'scripts/build_unified.sh')
        shutil.copy2(ROOT / 'scripts/packaging/resolve_torch_npu_wheel.py',
                     self.root / 'scripts/packaging/resolve_torch_npu_wheel.py')
        self.write('bin/git', '#!/bin/bash\nprintf "git %s\\n" "$*" >> "$TRACE"\n', executable=True)
        self.write('torch.py', 'class _C:\n    _GLIBCXX_USE_CXX11_ABI = True\n')
        self.write('build.sh', '''#!/bin/bash
printf 'mindspore %s ABI=%s\n' "$*" "${_GLIBCXX_USE_CXX11_ABI:-unset}" >> "$TRACE"
[[ "${FAIL_STAGE:-}" != ms ]] || exit 17
mkdir -p output
printf native-ms > output/mindspore-test.whl
''')
        self.write('third_party/torch_npu/ci/build.sh', '''#!/bin/bash
printf 'torch_npu %s ABI=%s\n' "$*" "${_GLIBCXX_USE_CXX11_ABI:-unset}" >> "$TRACE"
[[ "${FAIL_STAGE:-}" != torch_npu ]] || exit 18
mkdir -p dist
printf native-torch_npu > dist/torch_npu-test.whl
''')
        self.write('scripts/packaging/merge_whl.py', '''import os, pathlib, sys
assert '' not in sys.argv[1:], 'Unexpected empty argument'
with open(os.environ['TRACE'], 'a') as f:
    f.write('merge ' + ' '.join(sys.argv[1:]) + '\\n')
assert pathlib.Path(sys.argv[-1]).read_text() == 'native-ms'
assert pathlib.Path(sys.argv[sys.argv.index('--output-dir') + 1]) == pathlib.Path(sys.argv[-1]).parent
assert pathlib.Path(sys.argv[sys.argv.index('--module-wheel') + 1]).read_text() == 'native-torch_npu'
''')
        self.env = dict(os.environ, PATH=str(self.root / 'bin') + ':' + os.environ['PATH'],
                        TRACE=str(self.root / 'trace'), MS_PYTHON_EXECUTABLE=sys.executable)
        self.env.pop('_GLIBCXX_USE_CXX11_ABI', None)
        self.env.pop('CONDA_PREFIX', None)
        self.env.pop('VIRTUAL_ENV', None)

    def write(self, name, contents, executable=False):
        path = self.root / name
        path.write_text(contents)
        if executable:
            path.chmod(0o755)

    def run_build(self, *args, cwd=None):
        trace_file = self.root / 'trace'
        trace_file.unlink(missing_ok=True)
        result = subprocess.run(['bash', str(self.root / 'scripts/build_unified.sh'), *args],
                                cwd=cwd or self.root, env=self.env, capture_output=True,
                                text=True, timeout=30, check=False)
        trace = trace_file.read_text() if trace_file.exists() else ''
        return result, trace

    def test_native_options_and_abi_are_separate(self):
        result, trace = self.run_build('--torch-npu-source', '--torch=2.13.0', '--disable_fxrt', '-e', 'ascend', '-j4')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('mindspore -e ascend -j4 ABI=unset', trace)
        self.assertIn('--torch=2.13.0 --disable_fxrt ABI=1', trace)
        self.assertLess(trace.index('mindspore '), trace.index('torch_npu '))
        self.assertLess(trace.index('torch_npu '), trace.index('merge '))

    def test_prepare_only_needs_no_python_or_build(self):
        self.env['MS_PYTHON_EXECUTABLE'] = '/no/python'
        result, trace = self.run_build('--prepare-only')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(trace, 'git submodule update --init --recursive -- third_party/torch_npu\n')

    def stub_official_download(self):
        """Replace curl alone; keep the selected interpreter and build routing real."""
        self.env['EXPECTED_URL'] = WHEEL_URL
        self.write('bin/curl', f'#!{sys.executable}\n' + '''import os, pathlib, sys
assert '--fail' in sys.argv and '--location' in sys.argv
assert sys.argv[sys.argv.index('--proto') + 1] == '=https'
assert sys.argv[-1] == os.environ['EXPECTED_URL']
destination = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])
assert destination.name == 'torch_npu-2.13.0+test-cp310-cp310-manylinux_2_28_aarch64.whl'
with open(os.environ['TRACE'], 'a', encoding='utf-8') as stream:
    stream.write('download ' + str(destination.parent) + '\\n')
if os.environ.get('FAIL_STAGE') == 'download':
    sys.exit(20)
destination.write_text('native-torch_npu')
''', executable=True)
        self.write('pip.py', 'raise AssertionError("Prebuilt mode must not use pip")\n')
        self.write('torch.py', 'raise AssertionError("Prebuilt mode must not import torch")\n')
        shutil.rmtree(self.root / 'third_party')

    def test_official_wheel_skips_source_and_abi_probe(self):
        """Download first, compile only MindSpore, merge, and remove temporary downloads."""
        self.stub_official_download()
        self.env['_GLIBCXX_USE_CXX11_ABI'] = '0'
        result, trace = self.run_build('--torch-npu-wheel=' + WHEEL_URL, '-j4')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('git ', trace)
        self.assertNotIn('torch_npu --python', trace)
        self.assertIn('mindspore -V 910b -e ascend -j4 ABI=0', trace)
        self.assertLess(trace.index('download '), trace.index('mindspore '))
        self.assertLess(trace.index('mindspore '), trace.index('merge '))
        self.assertFalse(Path(trace.splitlines()[0].removeprefix('download ')).exists())

    def test_prebuilt_failures_stop_and_clean_download(self):
        """A failed download or MindSpore build must never merge a stale wheel."""
        self.stub_official_download()
        for stage, code in (('download', 20), ('ms', 17)):
            with self.subTest(stage=stage):
                self.env['FAIL_STAGE'] = stage
                result, trace = self.run_build('--torch-npu-wheel=' + WHEEL_URL)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertNotIn('merge ', trace)
                if stage == 'download':
                    self.assertNotIn('mindspore ', trace)
                self.assertFalse(Path(trace.splitlines()[0].removeprefix('download ')).exists())

    def test_version_selection_uses_resolved_download_url(self):
        """Route automatic selection through the same download, native build and merge path."""
        self.stub_official_download()
        self.write('scripts/packaging/resolve_torch_npu_wheel.py',
                   'import os, sys\nassert sys.argv[1] == "2.13.0"\nprint(os.environ["EXPECTED_URL"])\n')
        for args in ((), ('--torch-npu-wheel=2.13.0',)):
            with self.subTest(args=args):
                result, trace = self.run_build(*args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('merge ', trace)
                self.assertNotIn('git ', trace)
        self.write('scripts/packaging/resolve_torch_npu_wheel.py', 'raise SystemExit(21)\n')
        result, trace = self.run_build('--torch-npu-wheel=2.13.0')
        self.assertEqual(result.returncode, 21)
        self.assertEqual(trace, '')

    def test_prebuilt_rejects_source_options_and_invalid_urls(self):
        """Reject ineffective source-only flags before any download or build."""
        for option in ('--torch-npu-source', '--prepare-only', '--clean', '--torch=2.13.0', '--disable_torchair',
                       '--disable_fxrt', '--disable_rpc', '--enable_lto', '--enable_pgo=1'):
            with self.subTest(option=option):
                result, trace = self.run_build('--torch-npu-wheel=' + WHEEL_URL, option)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('cannot be combined', result.stderr)
                self.assertEqual(trace, '')
        for url in ('', 'invalid-version', 'http://example.invalid/torch_npu.whl',
                    'https://example.invalid/package.tar.gz',
                    'https://example.invalid/torch-2.13.0-cp310-cp310-linux_aarch64.whl'):
            with self.subTest(url=url):
                result, trace = self.run_build('--torch-npu-wheel=' + url)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(result.stderr)
                self.assertEqual(trace, '')

    def test_empty_optional_arguments_reach_merge(self):
        """Empty native/merge options must not abort under nounset or add empty arguments."""
        native = self.root / 'third_party/torch_npu/ci/build.sh'
        native.write_text(native.read_text() + '\n[[ $# == 1 && "$1" == --python=* ]] || exit 19\n')
        for numpy_version in ('1.26.4', '2.0.0'):
            with self.subTest(numpy_version=numpy_version):
                self.write('numpy.py', f'__version__ = "{numpy_version}"\n')
                result, trace = self.run_build('--torch-npu-source')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('merge ', trace)
                self.assertEqual('--require numpy<2' in trace, numpy_version.startswith('1.'))

    def test_failed_native_build_never_merges(self):
        for stage, code in (('ms', 17), ('torch_npu', 18)):
            with self.subTest(stage=stage):
                self.env['FAIL_STAGE'] = stage
                result, trace = self.run_build('--torch-npu-source')
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertNotIn('merge ', trace)

    def test_incompatible_torch_abi_fails_before_compilation(self):
        self.env['_GLIBCXX_USE_CXX11_ABI'] = '0'
        result, trace = self.run_build('--torch-npu-source')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ABI override', result.stderr)
        self.assertNotIn('mindspore ', trace)

    def test_device_defaults_preserve_explicit_native_options(self):
        """Apply Ascend defaults only when the caller has not selected a device."""
        for args, expected in (
                ([], '-V 910b -e ascend'),
                (['-j4'], '-V 910b -e ascend -j4'),
                (['-V', '310', '-j4'], '-e ascend -V 310 -j4'),
                (['-V910b'], '-e ascend -V910b'),
                (['-e', 'ascend', '-j4'], '-e ascend -j4'),
                (['-egpu', '-V11.1'], '-egpu -V11.1'),
                (['-e', 'cpu'], '-e cpu')):
            with self.subTest(args=args):
                result, trace = self.run_build('--torch-npu-source', *args)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f'mindspore {expected} ABI=unset\n', trace)

    def test_help_never_builds_or_merges_stale_wheel(self):
        """Both help flags must exit before source preparation or native builds."""
        (self.root / 'output').mkdir()
        self.write('output/mindspore-test.whl', 'stale')
        self.env['MS_PYTHON_EXECUTABLE'] = '/no/python'
        for flag in ('-h', '--help'):
            with self.subTest(flag=flag):
                result, trace = self.run_build(flag)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('Usage:', result.stdout)
                self.assertEqual(trace, '')
                self.assertEqual((self.root / 'output/mindspore-test.whl').read_text(), 'stale')

    def test_relative_venv_python_survives_native_directory_changes(self):
        """Keep the selected venv active when native commands change directories."""
        caller = self.root / 'caller'
        caller.mkdir()
        venv = caller / 'venv'
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages', str(venv)],
                       check=True, capture_output=True, text=True, timeout=30)
        self.env['EXPECTED_PREFIX'] = str(venv)
        # Check the interpreter after both native builds change their working directories.
        check = '''
python -c 'import os, sys; assert sys.prefix == os.environ["EXPECTED_PREFIX"], sys.prefix' || exit 19
'''
        for filename in ('build.sh', 'third_party/torch_npu/ci/build.sh'):
            path = self.root / filename
            path.write_text(path.read_text() + 'mkdir -p nested\ncd nested\n' + check)
        self.write('third_party/torch_npu/setup.py',
                   'import os, sys\nassert sys.prefix == os.environ["EXPECTED_PREFIX"]\n')
        result, trace = self.run_build('--torch-npu-source', '--python=./venv/bin/python', '--clean', cwd=caller)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('merge ', trace)

    def test_python_lookup_failure_has_diagnostic(self):
        result, trace = self.run_build('--python=/no/python')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Python executable not found or not executable: /no/python', result.stderr)
        self.assertEqual(trace, '')

    def test_python_entry_failure_has_diagnostic(self):
        """Report unavailable or mismatched native Python entry points before building."""
        directory = self.root / 'selected'
        directory.mkdir()
        (directory / 'selected-python').symlink_to(sys.executable)
        for body, message in (('echo /wrong/environment', 'Python environment mismatch: python'),
                              ('exit 1', 'Cannot run Python entry point: python')):
            with self.subTest(body=body):
                self.write('selected/python', '#!/bin/bash\n' + body + '\n', executable=True)
                result, trace = self.run_build('--python=' + str(directory / 'selected-python'))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertNotIn('mindspore ', trace)


class OfficialWheelSelectionTest(unittest.TestCase):
    """Select official files independently of the CI host's glibc and architecture."""

    def test_python_architecture_and_release_selection(self):
        """Select ARM/x86 and Python variants, reject missing or ambiguous releases."""
        spec = importlib.util.spec_from_file_location(
            'resolver', ROOT / 'scripts/packaging/resolve_torch_npu_wheel.py')
        resolver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resolver)
        files = []
        for arch in ('aarch64', 'x86_64'):
            for python in ('cp310', 'cp311'):
                name = f'torch_npu-2.13.0-{python}-{python}-manylinux_2_28_{arch}.whl'
                files.append({'name': name, 'browser_download_url': 'https://example.invalid/' + name,
                              'type': 'attach'})
        files.extend([
            {'name': 'source.tar.gz', 'type': 'source'},
            {'name': 'torch_npu.whl.sha256', 'type': 'attach'},
            {'name': 'torch-2.13.0-cp310-cp310-manylinux_2_28_aarch64.whl', 'type': 'attach'},
        ])
        for item in files[:4]:
            parts = item['name'].split('-')
            arch = parts[-1].removeprefix('manylinux_2_28_').removesuffix('.whl')
            self.assertEqual(resolver.select_wheel({'assets': files}, '2.13.0', parts[2], parts[3], arch),
                             item['browser_download_url'])
        for version, python, arch in (('2.14.0', 'cp310', 'aarch64'), ('2.13.0', 'cp312', 'aarch64'),
                                      ('2.13.0', 'cp310', 'ppc64le')):
            with self.assertRaisesRegex(ValueError, 'found 0'):
                resolver.select_wheel({'assets': files}, version, python, python, arch)
        files[0]['type'] = 'source'
        with self.assertRaisesRegex(ValueError, 'found 0'):
            resolver.select_wheel({'assets': files}, '2.13.0', 'cp310', 'cp310', 'aarch64')
        files[0]['type'] = 'attach'
        files.append(dict(files[0], browser_download_url='https://example.invalid/alternative.whl'))
        with self.assertRaisesRegex(ValueError, 'found 2'):
            resolver.select_wheel({'assets': files}, '2.13.0', 'cp310', 'cp310', 'aarch64')


    def test_resolver_queries_gitcode_with_curl(self):
        """Resolve via verified HTTPS curl and preserve request failure diagnostics."""
        spec = importlib.util.spec_from_file_location(
            'resolver', ROOT / 'scripts/packaging/resolve_torch_npu_wheel.py')
        resolver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(resolver)
        release = {'assets': [{'name': 'torch_npu-2.13.0-cp310-cp310-manylinux_2_28_aarch64.whl',
                               'type': 'attach', 'browser_download_url': WHEEL_URL}]}
        response = subprocess.CompletedProcess([], 0, stdout=json.dumps(release))
        tag = type('Tag', (), {'interpreter': 'cp310', 'abi': 'cp310'})()
        with patch.object(resolver.subprocess, 'run', return_value=response) as curl, \
                patch.object(resolver, 'sys_tags', side_effect=lambda: iter([tag])), \
                patch.object(resolver.platform, 'machine', return_value='aarch64'):
            self.assertEqual(resolver.resolve('2.13.0'), WHEEL_URL)
            command = curl.call_args.args[0]
            self.assertEqual(command[0], 'curl')
            self.assertEqual(command[-1], 'https://api.gitcode.com/api/v5/repos/Ascend/pytorch/'
                             'releases/tags/v26.2.0-pytorch2.13.0')
            self.assertIn('--fail', command)
            self.assertNotIn('--insecure', command)
            self.assertTrue(curl.call_args.kwargs['check'])
            with self.assertRaisesRegex(ValueError, 'No GitCode release pinned'):
                resolver.resolve('2.14.0')
            self.assertEqual(curl.call_count, 1)
            curl.side_effect = subprocess.CalledProcessError(60, command, stderr='certificate verify failed')
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                resolver.resolve('2.13.0')
            self.assertIn('certificate verify failed', caught.exception.stderr)


if __name__ == '__main__':
    unittest.main()
