"""Validate unified wheel contents and independent framework execution."""
import argparse
import configparser
import fnmatch
import importlib.metadata
import importlib.util
import json
from email.parser import BytesParser
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

BUNDLE = 'mindspore/_bundled/torch_npu/'


def check_torch_npu_payload(names, components=None):
    """torch_npu-specific checks live in validation, not in the wheel merger."""
    assert any(fnmatch.fnmatch(name, 'torch_npu/_C.*.so') for name in names), 'torch_npu Python extension missing'
    for library in ('libtorch_npu.so', 'libop_plugin_atb.so'):
        assert 'torch_npu/lib/' + library in names, library
    artifacts = {'torchair': ('torch_npu/dynamo/torchair/', 'torch_npu/dynamo/torchair/core/_torchair*.so'),
                 'fxrt': ('torch_npu/fxrt/', 'torch_npu/fxrt/_fxrt_api*.so'),
                 'rpc': ('torch_npu/lib/libtensorpipe.so', 'torch_npu/lib/libtensorpipe.so*')}
    for component, (prefix, pattern) in artifacts.items():
        present = any(name.startswith(prefix) for name in names)
        selected = components[component] if components is not None else present
        assert present == selected, component + ' selection does not match wheel contents'
        if selected:
            assert any(fnmatch.fnmatch(name, pattern) for name in names), component + ' native module missing'


def check_wheel(filename):
    """Check the unified wheel's payload, metadata and entry points."""
    with zipfile.ZipFile(filename) as archive:
        names = archive.namelist()
        roots = {name.split('/')[0] for name in names}
        metadata_roots = [root for root in roots if root.endswith('.dist-info')]
        assert len(metadata_roots) == 1, metadata_roots
        assert not any(part.endswith('.dist-info') for name in names
                       for part in name.split('/')[1:]), 'Nested distribution metadata'
        assert {'mindspore', '_mindspore_torch_npu_loader.py'} <= roots, roots
        assert not {'_mindspore_torch_npu_path.py', 'mindspore_torch_npu.pth', '_mindspore_info.py'} & roots, roots
        assert not {'torch', 'functorch', 'torchgen', 'torch_npu', 'torchnpugen'} & roots, roots
        assert BUNDLE + 'torchnpugen/__init__.py' in names
        metadata_root = metadata_roots[0]
        metadata = BytesParser().parsebytes(archive.read(metadata_root + '/METADATA'))
        assert metadata['Name'] == 'mindspore', metadata['Name']
        components = json.loads(archive.read(metadata_root + '/components.json'))
        core = [component for component in components['components'] if component['name'] == 'mindspore']
        assert len(core) == 1, f'{filename}: Expected exactly one mindspore component, found {len(core)}'
        assert core[0]['version'] == metadata['Version'], (
            f"{filename}: MindSpore Core version mismatch: expected {metadata['Version']}, got {core[0]['version']}")
        summary = 'MindSpore unified distribution; ' + '; '.join(
            f"{component['display_name']}={component['version']}" for component in components['components'])
        assert metadata['Summary'] == summary, metadata['Summary']
        torch_requirements = [item for item in metadata.get_all('Requires-Dist', [])
                              if item.startswith('torch==')]
        assert len(torch_requirements) == 1 and 'extra == "torch-npu"' in torch_requirements[0], torch_requirements
        entries = configparser.ConfigParser()
        entries.read_string(archive.read(metadata_root + '/entry_points.txt').decode())
        assert entries['torch.backends']['torch_npu'] == '_mindspore_torch_npu_loader:autoload'
        assert 'msrun' in entries['console_scripts']
        assert entries['console_scripts']['msinfo'] == '_mindspore_torch_npu_loader:msinfo'
        assert entries['console_scripts']['torch_npu_run'] == '_mindspore_torch_npu_loader:run'
        assert 'mindspore-torch-npu-run' not in entries['console_scripts']
        assert 'npu' in entries['torch_dynamo_backends']
        check_torch_npu_payload([name.removeprefix(BUNDLE) for name in names if name.startswith(BUNDLE)])
        assert 'Root-Is-Purelib: false' in archive.read(metadata_root + '/WHEEL').decode()
    print(f'Wheel structure OK: {Path(filename).name}')


def runtime_components(distribution):
    """Inspect the selected provider's optional components without importing frameworks."""
    spec = importlib.util.find_spec('torch_npu')
    if spec is None or spec.loader is None:
        # Inspect bundled components without triggering PyTorch/backend loading.
        spec = importlib.machinery.PathFinder.find_spec(
            'torch_npu', [str(distribution.locate_file(BUNDLE))])
    names = ['torch_npu/' + str(path.relative_to(root))
             for root in (map(Path, spec.submodule_search_locations or []) if spec else [])
             for path in root.rglob('*') if path.is_file()]
    return {
        name: any(path.startswith(prefix) for path in names) for name, prefix in (
            ('torchair', 'torch_npu/dynamo/torchair/'), ('fxrt', 'torch_npu/fxrt/'),
            ('rpc', 'torch_npu/lib/libtensorpipe.so'))}


def check_runtime(device, require_no_torch, shared_ascend=False, torch_npu_source=None):
    """Verify framework imports and execution in separate Python processes."""
    distribution = importlib.metadata.distribution('mindspore')
    components = runtime_components(distribution)
    mindspore = '''
import mindspore as ms
import numpy as np
ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
x = ms.Tensor(np.array([1., 2., 3.], dtype=np.float32))
np.testing.assert_allclose((x + x).asnumpy(), [2., 4., 6.])
'''
    torch = '''
import torch
assert 'torch_npu' in sys.modules, 'torch_npu backend entry point was not loaded'
x = torch.tensor([1., 2., 3.], requires_grad=True)
y = (x * x).sum()
y.backward()
assert y.item() == 14.
assert x.grad.tolist() == [2., 4., 6.]
'''
    cases = {
        'mindspore only': mindspore + "assert 'torch' not in sys.modules\nassert 'torch_npu' not in sys.modules\n",
        'torch only': torch + "assert 'mindspore' not in sys.modules\n",
        'torch then explicit torch_npu': torch + 'import torch_npu\n' + "assert 'mindspore' not in sys.modules\n",
        'mindspore then torch': mindspore + torch,
        'torch then mindspore': torch + mindspore,
    }
    compiler_imports = ''
    if components['torchair']:
        compiler_imports += 'from torch_npu.dynamo import torchair\n'
    if components['fxrt']:
        compiler_imports += 'import torch_npu.fxrt\n'
    if compiler_imports:
        cases['torch_npu compiler modules'] = torch + compiler_imports + "assert 'mindspore' not in sys.modules\n"
    if require_no_torch:
        cases = {'mindspore without torch installed':
                 "import importlib.util\nassert importlib.util.find_spec('torch') is None\n" + cases['mindspore only']}
    if device is not None:
        torch_ascend = f'''
import torch
torch.npu.set_device({device})
x = torch.tensor([1., 2., 3.], device='npu', requires_grad=True)
y = (x * x).sum()
y.backward()
assert y.item() == 14.
assert x.grad.cpu().tolist() == [2., 4., 6.]
'''
        mindspore_ascend = f'''
import numpy as np
import mindspore as ms
ms.set_context(mode=ms.PYNATIVE_MODE, device_target='Ascend', device_id={device})
def square_sum(value):
    return (value * value).sum()
x = ms.Tensor(np.array([1., 2., 3.], dtype=np.float32))
y, grad = ms.value_and_grad(square_sum)(x)
np.testing.assert_allclose(y.asnumpy(), 14.)
np.testing.assert_allclose(grad.asnumpy(), [2., 4., 6.])
'''
        cases['mindspore Ascend forward/backward'] = mindspore_ascend + "assert 'torch' not in sys.modules\n"
        if not require_no_torch:
            cases['torch Ascend forward/backward'] = torch_ascend + "assert 'mindspore' not in sys.modules\n"
            if shared_ascend:
                cases['shared Ascend runtimes'] = torch_ascend + mindspore_ascend
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='1', PYTHONNOUSERSITE='1')
    source_check = ''
    if torch_npu_source:
        bundle = str(distribution.locate_file(BUNDLE).resolve())
        source_check = f'''
if 'torch_npu' in sys.modules:
    from pathlib import Path
    module = sys.modules['torch_npu']
    is_bundled = Path(module.__file__).resolve().is_relative_to({bundle!r})
    assert is_bundled == {torch_npu_source == 'bundled'}, module.__file__
    print('torch_npu source:', module.__file__)
'''
    # Retain SDK Python modules, without exposing framework source directories.
    sdk_root = Path(env.get('ASCEND_HOME_PATH', '/nonexistent-ascend-sdk')).resolve()
    env['PYTHONPATH'] = os.pathsep.join(
        path for path in env.get('PYTHONPATH', '').split(os.pathsep)
        if path and Path(path).resolve().is_relative_to(sdk_root))
    with tempfile.TemporaryDirectory(prefix='mindspore-unified-check-') as directory:
        for name, code in cases.items():
            subprocess.run([sys.executable, '-c', 'import sys\n' + code + source_check], env=env,
                           cwd=directory, check=True, timeout=300)
            print(f'Runtime OK: {name}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel')
    parser.add_argument('--module-wheel')
    parser.add_argument('--runtime', action='store_true')
    parser.add_argument('--device', type=int)
    parser.add_argument('--require-no-torch', action='store_true')
    parser.add_argument('--shared-ascend', action='store_true')
    parser.add_argument('--torch-npu-source', choices=('bundled', 'external'),
                        help='Assert which torch_npu is imported')
    args = parser.parse_args()
    if not args.wheel and not args.runtime and not args.module_wheel:
        parser.error('choose --wheel and/or --runtime')
    if args.shared_ascend and (args.device is None or args.require_no_torch):
        parser.error('--shared-ascend requires --device and an installed PyTorch')
    if args.module_wheel:
        with zipfile.ZipFile(args.module_wheel) as module_archive:
            check_torch_npu_payload(module_archive.namelist())
        print('Native torch_npu wheel contents OK')
    if args.wheel:
        check_wheel(args.wheel)
    if args.runtime:
        check_runtime(args.device, args.require_no_torch, args.shared_ascend, args.torch_npu_source)
