"""Exercise the public merge command with independently packaged backend fixtures."""
from email.parser import BytesParser
import base64
import csv
import hashlib
import io
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest
import zipfile

from packaging.version import Version


ROOT = Path(__file__).resolve().parents[4]
COMMAND = ROOT / 'scripts/build/merge_whl_package.sh'
PYTAG = f'cp{sys.version_info.major}{sys.version_info.minor}'
PLATFORM = sysconfig.get_platform().replace('-', '_').replace('.', '_')


class MergeWheelTest(unittest.TestCase):
    """Validate wheel merging and backend loading with synthetic native packages."""

    def setUp(self):
        self.work = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.work)
        self.output = self.work / 'output'
        self.output.mkdir()
        (self.output / 'keep.txt').write_text('existing output')
        self.version_before = (ROOT / 'version.txt').read_bytes()

    def wheel(self, name, *, torch_npu=False, legacy=False, gpu=None, version='3.0.0beta',
              tag=PYTAG, platform_tag=PLATFORM, torch_npu_version='2.13.0', torch_npu_binary=b'torch_npu',
              omit=None, native=False):
        """Create an independently packaged wheel with selectable metadata and payload."""
        directory = self.work / name
        directory.mkdir()
        normalized = str(Version(torch_npu_version if native else version))
        distribution = 'torch_npu' if native else 'mindspore'
        filename = directory / f'{distribution}-{normalized}-{tag}-{tag}-{platform_tag}.whl'
        module = 'mindspore.py310' if legacy else 'mindspore'
        payload = {
            'mindspore/__init__.py': b'from .version import __version__\n',
            'mindspore/version.py': f"__version__ = '{version}'\n".encode(),
            'mindspore/.commit_id': b"__commit_id__ = ''input-commit''\n",
            'mindspore/lib/libakg.so': b'old akg',
            f'mindspore/_c_dataengine.{tag}.so': b'engine' * (8 if torch_npu else 1),
        }
        if gpu:
            payload[f'mindspore/lib/plugin/gpu{gpu}/libcuda_ops.so.{gpu}'] = gpu.encode()
            payload[f'mindspore/lib/plugin/libmindspore_gpu.so.{gpu}'] = gpu.encode()
        if torch_npu:
            payload.update({
                'mindspore/lib/plugin/libmindspore_ascend.so.2': b'ascend',
                'mindspore/lib/libmindspore_ge_backend.so': b'GE',
                'torch_npu/__init__.py': b'',
                'torch_npu/version.py': f"__version__ = '{torch_npu_version}'\n".encode(),
                'torch_npu/_C.test.so': torch_npu_binary,
                'torch_npu/bin/helper': b'#!/bin/sh\nexit 0\n',
                'torch_npu/lib/libtorch_npu.so': b'libtorch_npu',
                'torch_npu/lib/libop_plugin_atb.so': b'ATB',
                'torchnpugen/__init__.py': b'',
                'torch_npu/dynamo/torchair/core/_torchair.so': b'torchair',
                'torch_npu/fxrt/_fxrt_api.so': b'fxrt',
                'torch_npu/lib/libtensorpipe.so': b'rpc',
            })
            entries_torch_npu = {
                'console_scripts': ['msrun=mindspore.parallel.cluster.run:main',
                                    'torch_npu_run=torch_npu.distributed.run:_main'],
                'torch.backends': ['torch_npu=torch_npu:_autoload'],
                'torch.distributed.backends': ['hccl=torch_npu:register'],
                'torch_dynamo_backends': ['npu=torch_npu:backend']}
        if omit:
            del payload[omit]
        if native:
            payload = {name: data for name, data in payload.items() if not name.startswith('mindspore/')}
            payload['future_module/resource.bin'] = b'future component'
        metadata = f'{distribution}-{normalized}.dist-info'
        payload[metadata + '/METADATA'] = (
            f'Metadata-Version: 2.1\nName: {distribution}\nVersion: {normalized}\n'.encode())
        entries = '' if torch_npu else (
            '[console_scripts]\nmsrun=mindspore.parallel.cluster.run:main\nbase-helper=mindspore:helper\n')
        if not torch_npu:
            payload[metadata + '/METADATA'] += (
                b'Provides-Extra: base\nRequires-Dist: base-dependency; extra == "base"\n')
        if torch_npu:
            payload[metadata + '/METADATA'] += (
                f'Provides-Extra: torch-npu\nRequires-Dist: torch=={torch_npu_version}.*; '
                'extra == "torch-npu"\nRequires-Dist: pyyaml\n').encode()
            entries += '\n'.join(f'\n[{group}]\n' + '\n'.join(values)
                                 for group, values in entries_torch_npu.items())
        if native:
            payload[metadata + '/METADATA'] = payload[metadata + '/METADATA'].replace(
                b'Provides-Extra: torch-npu\n', b'').replace(b'; extra == "torch-npu"', b'')
            payload[metadata + '/METADATA'] = payload[metadata + '/METADATA'].replace(
                b'Metadata-Version: 2.1', b'Metadata-Version: 2.4')
            payload[metadata + '/METADATA'] += b'License-File: LICENSE\n'
            payload[metadata + '/licenses/LICENSE'] = b'upstream license'
        payload[metadata + '/top_level.txt'] = b'torch_npu\ntorchnpugen\n' if native else b'mindspore\n'
        payload[metadata + '/entry_points.txt'] = entries.encode()
        payload[metadata + '/WHEEL'] = f'Wheel-Version: 1.0\nTag: {tag}-{tag}-{platform_tag}\n'.encode()
        self.write_wheel(filename, payload, module)
        return filename

    @staticmethod
    def write_wheel(filename, payload, module):
        """Write payload bytes with legacy package paths and executable helper permissions."""
        with zipfile.ZipFile(filename, 'w') as archive:
            for path, data in payload.items():
                destination = path.replace('mindspore/', module + '/', 1) if path.startswith('mindspore/') else path
                info = zipfile.ZipInfo(destination)
                info.external_attr = (0o100755 if path == 'torch_npu/bin/helper' else 0o100644) << 16
                archive.writestr(info, data)

    def run_merge(self, *wheels, succeeds=True, module=None, requirements=()):
        """Run the public merger and check its output or failure isolation."""
        expected_filename = wheels[0].name
        previous = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.output.glob('*.whl')}
        env = dict(os.environ, PYTHON=sys.executable)
        # A caller's unrelated build variables must not change artifact metadata.
        env['_GLIBCXX_USE_CXX11_ABI'] = 'wrong-host-abi'
        result = subprocess.run(['bash', str(COMMAND), '--output-dir', str(self.output),
                                 *map(str, wheels), *(['--module-wheel', str(module)] if module else []),
                                 *[arg for value in requirements for arg in ('--require', value)]],
                                cwd=self.work, env=env,
                                capture_output=True, text=True, timeout=90, check=False)
        self.assertEqual((ROOT / 'version.txt').read_bytes(), self.version_before)
        self.assertEqual((self.output / 'keep.txt').read_text(), 'existing output')
        if not succeeds:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertEqual(previous, {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in self.output.glob('*.whl')})
            return result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        wheels = list(self.output.glob('*.whl'))
        self.assertEqual(len(wheels), 1)
        wheel = wheels[0]
        self.assertEqual(wheel.name, expected_filename)
        self.assertEqual(Path(str(wheel) + '.sha256').read_text(encoding='utf-8').split()[0],
                         hashlib.sha256(wheel.read_bytes()).hexdigest())
        return wheel

    def test_torch_npu_in_base_with_multiple_gpu_versions_and_legacy_layout(self):
        """Merge legacy GPU layouts while preserving the base wheel's torch_npu payload."""
        wheel = self.run_merge(self.wheel('ascend with torch_npu', torch_npu=True),
                               self.wheel('cuda119', gpu='11.9', legacy=True),
                               self.wheel('cuda1110', gpu='11.10'))
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            self.assertIn('mindspore/lib/plugin/gpu/libcuda_ops.so.11.9', names)
            self.assertIn('mindspore/lib/plugin/gpu/libcuda_ops.so.11.10', names)
            self.assertNotIn('mindspore/lib/plugin/gpu11.9/libcuda_ops.so.11.9', names)
            self.assertEqual(archive.read('mindspore/_bundled/torch_npu/torch_npu/_C.test.so'), b'torch_npu')
            helper_mode = archive.getinfo('mindspore/_bundled/torch_npu/torch_npu/bin/helper').external_attr >> 16
            self.assertTrue(helper_mode & 0o111)
            self.assertNotIn('mindspore/lib/libakg.so', names)
            metadata_files = [n for n in names if n.endswith('.dist-info/METADATA')]
            self.assertEqual(len(metadata_files), 1)
            metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
            self.assertEqual(metadata['Version'], '3.0.0b0')
            self.assertTrue(any(r.startswith('torch==') and 'extra == "torch-npu"' in r
                                for r in metadata.get_all('Requires-Dist')))

    def test_torch_npu_in_supplemental_wheel_preserves_all_components(self):
        """Retain backend libraries, module components and metadata from supplemental wheels."""
        wheel = self.run_merge(self.wheel('gpu base', gpu='12.0'), self.wheel('ascend', torch_npu=True))
        with zipfile.ZipFile(wheel) as archive:
            self.assertEqual(archive.read(f'mindspore/_c_dataengine.{PYTAG}.so'), b'engine' * 8)
            info = next(n.removesuffix('METADATA') for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
            metadata = BytesParser().parsebytes(archive.read(info + 'METADATA'))
            self.assertIn('base', metadata.get_all('Provides-Extra'))
            self.assertIn('base-dependency; extra == "base"', metadata.get_all('Requires-Dist'))
            self.assertIn('base-helper', archive.read(info + 'entry_points.txt').decode())
            self.assertEqual(archive.read('mindspore/lib/libmindspore_ge_backend.so'), b'GE')
            self.assertEqual(archive.read('mindspore/_bundled/torch_npu/torch_npu/fxrt/_fxrt_api.so'), b'fxrt')
            self.assertEqual(archive.read('mindspore/_bundled/torch_npu/torch_npu/dynamo/torchair/core/_torchair.so'),
                             b'torchair')
            self.assertEqual(archive.read('mindspore/_bundled/torch_npu/torch_npu/lib/libtensorpipe.so'), b'rpc')

    def test_rejects_missing_torch_npu(self):
        message = self.run_merge(self.wheel('no torch_npu', gpu='11.8'), succeeds=False)
        self.assertIn('At least one input wheel must contain torch_npu', message)

    def test_native_module_has_no_custom_manifest_and_preserves_future_files(self):
        """Preserve the complete native payload and expose only the unified distribution."""
        native = self.wheel('native', torch_npu=True, native=True)
        wheel = self.run_merge(self.wheel('base'), module=native)
        with zipfile.ZipFile(wheel) as archive:
            self.assertEqual(archive.read('mindspore/_bundled/torch_npu/future_module/resource.bin'),
                             b'future component')
            info = next(n.removesuffix('METADATA') for n in archive.namelist() if n.endswith('.dist-info/METADATA'))
            metadata = BytesParser().parsebytes(archive.read(info + 'METADATA'))
            self.assertIn('torch==2.13.0.*; extra == "torch-npu"', metadata.get_all('Requires-Dist'))
            self.assertEqual(archive.read(info + 'modules/torch-npu/licenses/LICENSE'), b'upstream license')
            self.assertEqual(metadata.get_all('License-File'), ['torch-npu/LICENSE'])
            self.assertEqual(archive.read(info + 'licenses/torch-npu/LICENSE'), b'upstream license')
            self.assertEqual(set(archive.read(info + 'top_level.txt').decode().splitlines()),
                             {'mindspore', '_mindspore_torch_npu_loader'})
            self.assertEqual(len([n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]), 1)
            self.assertFalse(any(n.startswith(('torch_npu/', 'torchnpugen/')) for n in archive.namelist()))
            entries = archive.read(info + 'entry_points.txt').decode()
            self.assertIn('torch_npu_run = _mindspore_torch_npu_loader:run', entries)
            self.assertNotIn('mindspore-torch-npu-run', entries)
            self.assertIn('torch_npu = _mindspore_torch_npu_loader:autoload', entries)
            self.assertIn('_mindspore_torch_npu_loader.py', archive.namelist())
            self.assertNotIn('mindspore_torch_npu.pth', archive.namelist())
            self.assertNotIn('_mindspore_torch_npu_path.py', archive.namelist())

    def test_torch_hook_prefers_external_torch_npu_in_either_backend_order(self):
        """Select the external provider independently of backend entry-point ordering."""
        wheel = self.run_merge(self.wheel('base'), module=self.wheel('native', torch_npu=True, native=True))
        installed = self.work / 'site-packages'
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(installed)
        # Model the native import/entry-point protocol without loading hardware libraries.
        (installed / 'mindspore/__init__.py').write_text("raise AssertionError('MindSpore must not initialize')\n")
        torch = installed / 'torch'
        torch.mkdir()
        (torch / '__init__.py').write_text('''import os
from importlib.metadata import entry_points
if os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD', '1') == '1':
    backends = list(entry_points(group='torch.backends'))
    if os.environ.get('TEST_REVERSE_BACKENDS') == '1':
        backends.reverse()
    for entry in backends:
        entry.load()()
''')
        bundled = installed / 'mindspore/_bundled/torch_npu'
        (installed / 'torch_npu/_inductor/__pycache__').mkdir(parents=True)
        (installed / 'torch_npu/_inductor/__pycache__/old.pyc').write_bytes(b'uninstall residue')
        external = self.work / 'external-site'
        external.mkdir()
        torch_npu_init = '''import os, builtins
previous = os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD', '1')
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch
builtins.torch_npu_initializations = getattr(builtins, 'torch_npu_initializations', 0) + 1
SOURCE = {source!r}
def _autoload():
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = previous
_autoload()
'''
        (bundled / 'torch_npu/__init__.py').write_text(torch_npu_init.format(source='bundled'))
        for source in ('bundled', 'external'):
            if source == 'external':
                (external / 'torch_npu').mkdir()
                (external / 'torch_npu/__init__.py').write_text(torch_npu_init.format(source='external'))
                info = external / 'torch_npu-2.13.0.dist-info'
                info.mkdir()
                (info / 'METADATA').write_text('Name: torch-npu\nVersion: 2.13.0\n')
                (info / 'entry_points.txt').write_text('[torch.backends]\ntorch_npu=torch_npu:_autoload\n')
            for reverse in ('0', '1'):
                code = f'''import site, sys, builtins
import os
site.addsitedir({str(installed)!r})
assert not {{'mindspore', 'torch', 'torch_npu'}} & sys.modules.keys()
sys.path.append({str(external)!r})
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '1'
os.environ['TEST_REVERSE_BACKENDS'] = {reverse!r}
import torch
import torch_npu
assert torch_npu.SOURCE == {source!r}
assert builtins.torch_npu_initializations == 1
assert 'mindspore' not in sys.modules
assert 'mindspore.torch_npu' not in sys.modules
assert ({str(bundled)!r} in sys.path) == {source == 'bundled'}
'''
                result = subprocess.run([sys.executable, '-S', '-c', code], cwd=self.work,
                                        capture_output=True, text=True, timeout=30, check=False)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        # An incomplete external provider must not silently borrow bundled helpers.
        code = f'''import site, sys
site.addsitedir({str(installed)!r})
sys.path.append({str(external)!r})
import torch
import torch_npu
import torchnpugen
'''
        result = subprocess.run([sys.executable, '-S', '-c', code], cwd=self.work,
                                capture_output=True, text=True, timeout=30, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No module named 'torchnpugen'", result.stderr)

    def test_no_startup_hook_and_explicit_loading_when_autoload_is_disabled(self):
        """Avoid startup hooks while allowing the launcher to load the bundled provider."""
        wheel = self.run_merge(self.wheel('base'), module=self.wheel('native', torch_npu=True, native=True))
        installed = self.work / 'site-packages'
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(installed)
        bundled = installed / 'mindspore/_bundled/torch_npu'
        (installed / 'torch').mkdir()
        (installed / 'torch/__init__.py').write_text('')
        (bundled / 'torch_npu/__init__.py').write_text('def _autoload(): pass\n')
        (bundled / 'torch_npu/distributed').mkdir()
        (bundled / 'torch_npu/distributed/__init__.py').write_text('')
        (bundled / 'torch_npu/distributed/run.py').write_text('def _main(): return 23\n')
        code = f'''import site, sys
before = list(sys.meta_path)
site.addsitedir({str(installed)!r})
assert not {{'pathlib', 'mindspore', 'torch', 'torch_npu', '_mindspore_torch_npu_loader'}} & sys.modules.keys()
assert sys.meta_path == before
from importlib.util import find_spec
assert find_spec('torch_npu') is None
import fractions
assert '_mindspore_torch_npu_loader' not in sys.modules
import torch
assert 'torch_npu' not in sys.modules
from _mindspore_torch_npu_loader import run
assert run() == 23
import torch_npu
import future_module
assert 'mindspore' not in sys.modules
assert sys.meta_path == before
'''
        result = subprocess.run([sys.executable, '-S', '-c', code], cwd=self.work,
                                capture_output=True, text=True, timeout=30, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_merger_runs_without_a_source_checkout(self):
        script = self.work / 'merge.py'
        script.write_bytes((ROOT / 'scripts/packaging/merge_whl.py').read_bytes())
        base = self.wheel('base')
        module = self.wheel('native', torch_npu=True, native=True)
        result = subprocess.run([sys.executable, str(script), str(base), '--module-wheel', str(module)],
                                cwd=self.work, capture_output=True, text=True, timeout=90, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.output / base.name).is_file())

    def test_merge_python_does_not_have_to_match_input_wheels(self):
        other_tag = 'cp311' if PYTAG != 'cp311' else 'cp310'
        wheel = self.run_merge(self.wheel('base', tag=other_tag),
                               module=self.wheel('native', torch_npu=True, native=True, tag=other_tag))
        self.assertIn(f'-{other_tag}-{other_tag}-', wheel.name)

    def test_explicit_build_requirement_is_preserved(self):
        wheel = self.run_merge(self.wheel('base'), module=self.wheel('native', torch_npu=True, native=True),
                               requirements=['numpy<2'])
        with zipfile.ZipFile(wheel) as archive:
            info = next(name for name in archive.namelist() if name.endswith('.dist-info/METADATA'))
            self.assertIn('numpy<2', BytesParser().parsebytes(archive.read(info)).get_all('Requires-Dist'))

    def test_merge_replaces_native_wheel_in_its_output_directory(self):
        """Keep the native artifact on validation failure, then replace it after successful packing."""
        base = self.wheel('native output')
        self.output = base.parent
        (self.output / 'keep.txt').write_text('existing output', encoding='utf-8')
        checksum = Path(str(base) + '.sha256')
        checksum.write_text(hashlib.sha256(base.read_bytes()).hexdigest() + '  ' + base.name + '\n', encoding='utf-8')
        original_checksum = checksum.read_bytes()
        bad = self.wheel('bad module', torch_npu=True, native=True, tag='cp999')
        self.assertIn('Incompatible module', self.run_merge(base, module=bad, succeeds=False))
        self.assertEqual(checksum.read_bytes(), original_checksum)
        module = self.wheel('official module', torch_npu=True, native=True)
        result = self.run_merge(base, module=module)
        self.assertEqual(result, base)
        self.assertNotEqual(checksum.read_bytes(), original_checksum)
        self.assertFalse((self.output / 'unified').exists())
        self.assertFalse(any(path.is_dir() for path in self.output.iterdir()))
        with zipfile.ZipFile(result) as archive:
            self.assertIn('mindspore/_bundled/torch_npu/torch_npu/lib/libtorch_npu.so', archive.namelist())

    def test_native_component_selection_and_tags(self):
        native = self.wheel('without fxrt', torch_npu=True, native=True, omit='torch_npu/fxrt/_fxrt_api.so')
        wheel = self.run_merge(self.wheel('base'), module=native)
        with zipfile.ZipFile(wheel) as archive:
            self.assertNotIn('mindspore/_bundled/torch_npu/torch_npu/fxrt/_fxrt_api.so', archive.namelist())
        bad = self.wheel('wrong module abi', torch_npu=True, native=True, tag='cp999')
        self.assertIn('Incompatible module', self.run_merge(self.wheel('second base'), module=bad, succeeds=False))

    def test_official_manylinux_module_keeps_generic_linux_tag(self):
        """Preserve MindSpore tags while retaining the official module's native payload."""
        architecture = platform.machine()
        base = self.wheel('linux base', platform_tag=f'linux_{architecture}')
        module = self.wheel('official', torch_npu=True, native=True,
                            platform_tag=f'manylinux_2_28_{architecture}.manylinux_2_34_{architecture}')
        wheel = self.run_merge(base, module=module)
        self.assertTrue(wheel.name.endswith(f'-linux_{architecture}.whl'))
        with zipfile.ZipFile(wheel) as archive:
            info = next(name for name in archive.namelist() if name.endswith('.dist-info/WHEEL'))
            metadata = BytesParser().parsebytes(archive.read(info))
            self.assertEqual(metadata.get_all('Tag'), [f'{PYTAG}-{PYTAG}-linux_{architecture}'])
            self.assertIn('mindspore/_bundled/torch_npu/torch_npu/lib/libtorch_npu.so', archive.namelist())

    def test_official_wheel_tag_mismatches_are_rejected(self):
        """Do not admit a different architecture, Python ABI, libc, or manylinux baseline."""
        architecture = platform.machine()
        other_architecture = 'x86_64' if architecture == 'aarch64' else 'aarch64'
        for index, (base_platform, module_platform, python_tag) in enumerate((
                (f'linux_{architecture}', f'manylinux_2_28_{other_architecture}', PYTAG),
                (f'linux_{architecture}', f'manylinux_2_28_{architecture}', 'cp999'),
                (f'linux_{architecture}', f'musllinux_1_2_{architecture}', PYTAG),
                (f'manylinux_2_17_{architecture}', f'manylinux_2_28_{architecture}', PYTAG))):
            with self.subTest(platform=module_platform, python=python_tag):
                base = self.wheel(f'base {index}', platform_tag=base_platform)
                module = self.wheel(f'module {index}', torch_npu=True, native=True,
                                    platform_tag=module_platform, tag=python_tag)
                self.assertIn('Incompatible module', self.run_merge(base, module=module, succeeds=False))

    def test_rejects_conflicting_torch_npu_versions_and_native_files(self):
        first = self.wheel('first', torch_npu=True)
        message = self.run_merge(first, self.wheel('other version', torch_npu=True, torch_npu_version='2.14.0'),
                                 succeeds=False)
        self.assertIn('Conflicting torch_npu builds', message)
        message = self.run_merge(first, self.wheel('other native build', torch_npu=True,
                                                 torch_npu_binary=b'other torch_npu'), succeeds=False)
        self.assertIn('Conflicting torch_npu builds', message)

    def test_single_backend_is_idempotent_with_separate_or_embedded_torch_npu(self):
        """Repacking a single backend must preserve native libraries and wheel contents."""
        native = self.wheel('native', torch_npu=True, native=True)
        wheel = self.run_merge(self.wheel('base', gpu='12.0'), module=native)
        with zipfile.ZipFile(wheel) as archive:
            before = {name: archive.read(name) for name in archive.namelist()}
        self.assertIn('mindspore/lib/libakg.so', before)
        self.assertIn('mindspore/lib/plugin/gpu12.0/libcuda_ops.so.12.0', before)
        self.assertNotIn('mindspore/lib/plugin/gpu/libcuda_ops.so.12.0', before)
        for module in (native, None):
            with self.subTest(module_wheel=module is not None):
                wheel = self.run_merge(wheel, module=module)
                with zipfile.ZipFile(wheel) as archive:
                    self.assertEqual(before, {name: archive.read(name) for name in archive.namelist()})

    def test_identical_torch_npu_providers_are_allowed(self):
        self.run_merge(self.wheel('first', torch_npu=True), self.wheel('same', torch_npu=True, gpu='11.8'))

    def test_preserves_compressed_manylinux_platform_tags(self):
        """Keep every input platform tag and regenerate valid RECORD hashes."""
        tag = f'manylinux_2_27_{platform.machine()}.manylinux_2_28_{platform.machine()}'
        wheel = self.run_merge(self.wheel('ascend', torch_npu=True, platform_tag=tag),
                              self.wheel('gpu', gpu='11.8', platform_tag=tag))
        with zipfile.ZipFile(wheel) as archive:
            record = next(name for name in archive.namelist() if name.endswith('.dist-info/RECORD'))
            for name, checksum, size in csv.reader(io.StringIO(archive.read(record).decode())):
                if name == record:
                    continue
                data = archive.read(name)
                self.assertEqual(len(data), int(size))
                expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                self.assertEqual(checksum, 'sha256=' + expected)
            metadata = BytesParser().parsebytes(archive.read(record.removesuffix('RECORD') + 'WHEEL'))
            self.assertEqual(set(metadata.get_all('Tag')), {f'{PYTAG}-{PYTAG}-{p}' for p in tag.split('.')})

    def test_rejects_different_mindspore_versions_and_python_tags(self):
        first = self.wheel('first', torch_npu=True)
        message = self.run_merge(first, self.wheel('old', version='2.12.0'), succeeds=False)
        self.assertIn('Incompatible MindSpore version or wheel tags', message)
        message = self.run_merge(first, self.wheel('wrong python', tag='cp999'), succeeds=False)
        self.assertIn('Incompatible MindSpore version or wheel tags', message)


if __name__ == '__main__':
    unittest.main()
