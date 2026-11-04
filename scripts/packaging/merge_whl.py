"""Merge MindSpore backend wheels while retaining the complete torch_npu module."""
import argparse
import ast
import configparser
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

try:
    from setuptools.command.bdist_wheel import WheelFile
except ImportError:
    from wheel.wheelfile import WheelFile
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version
from packaging.requirements import Requirement
from packaging.markers import Marker


BACKEND_LIBRARIES = ('libmpi_collective.so', 'libmpi_adapter.so',
                     'libmindspore.so', 'libmindspore_ge_backend.so')
BUNDLE = Path('mindspore/_bundled/torch_npu')
# Embedded so this merger remains usable without a source checkout or companion files.
TORCH_NPU_LOADER = '''"""Load torch_npu and report installed MindSpore component versions."""


def autoload():
    from importlib.util import find_spec
    import os
    import sys

    spec = find_spec("torch_npu")
    if spec is None or spec.loader is None:
        root = os.path.realpath(os.path.join(os.path.dirname(__file__), "mindspore/_bundled/torch_npu"))
        if root not in sys.path:
            sys.path.append(root)
    import torch_npu
    torch_npu._autoload()


def run():
    import torch
    autoload()  # The launcher also works when backend autoloading is disabled.
    from torch_npu.distributed.run import _main
    return _main()


def msinfo():
    """Print the component information recorded when the wheel was built."""
    import argparse
    from importlib.metadata import PackageNotFoundError, distribution
    import json

    parser = argparse.ArgumentParser(
        prog='msinfo', description='Display installed MindSpore component versions without loading frameworks.')
    parser.parse_args()
    try:
        package = distribution('mindspore')
    except PackageNotFoundError:
        parser.exit(1, 'msinfo: MindSpore is not installed in this Python environment.\\n')
    try:
        contents = package.read_text('components.json')
        if contents is None:
            parser.exit(1, 'msinfo: This MindSpore installation has no component information.\\n')
        components = json.loads(contents)['components']
        if not isinstance(components, list) or not components:
            raise ValueError('expected a nonempty components list')
        for component in components:
            if not isinstance(component, dict) or any(
                    not isinstance(component.get(key), str) or not component[key].strip()
                    for key in ('name', 'display_name', 'version')):
                raise ValueError('each component requires nonempty name, display_name and version strings')
    except (ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'msinfo: Invalid components.json: {error}. Reinstall MindSpore.\\n')
    print('MindSpore:')
    print(f'  Version: {package.version}')
    print('  Components:')
    for component in components:
        print(f"    {component['display_name']}:")
        print(f"      Version: {component['version']}")
    return 0
'''


def digest(path):
    """Hash large native libraries without reading them all into memory."""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_version(package, filename='mindspore/version.py'):
    """Read the input wheel's version without importing its native extensions."""
    tree = ast.parse((package / filename).read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == '__version__' for target in node.targets):
            return str(ast.literal_eval(node.value))
    raise ValueError(f'Missing __version__ in {package / filename}')


def unpack(filename, destination):
    """Extract a wheel and normalize the legacy mindspore.py* package directory."""
    _, _, _, tags = parse_wheel_filename(filename.name)
    with zipfile.ZipFile(filename) as archive:
        for member in archive.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError(f'Invalid wheel member: {member.filename}')
        archive.extractall(destination)
        for member in archive.infolist():
            path = destination / member.filename
            if path.is_file():
                path.chmod(path.stat().st_mode | ((member.external_attr >> 16) & 0o111))
    metadata_files = list(destination.glob('*.dist-info/METADATA'))
    if len(metadata_files) != 1:
        raise ValueError(f'{filename}: expected one distribution metadata directory')
    metadata = BytesParser().parsebytes(metadata_files[0].read_bytes())
    name = canonicalize_name(metadata['Name'] or '')
    if name == 'torch-npu':
        version = metadata['Version']
    elif name.startswith('mindspore'):
        if not (destination / 'mindspore').is_dir():
            legacy = list(destination.glob('mindspore.py*'))
            if len(legacy) != 1 or not legacy[0].is_dir():
                raise ValueError(f'{filename}: no MindSpore package found')
            legacy[0].rename(destination / 'mindspore')
        version = read_version(destination)
    else:
        raise ValueError(f'{filename}: expected a MindSpore or torch_npu wheel')
    if any((destination / name).exists() for name in ('torch', 'functorch', 'torchgen')):
        raise ValueError(f'{filename}: PyTorch must remain separately installed')
    entries = configparser.ConfigParser(interpolation=None)
    entries.optionxform = str
    entries.read(metadata_files[0].parent / 'entry_points.txt')
    return {'path': destination, 'filename': filename, 'tags': tags, 'name': name,
            'version': version, 'metadata': metadata, 'entries': entries,
            'info': metadata_files[0].parent}


def module_files(item):
    """The module owns its complete payload, including future packages and resources."""
    root = item['path'] / BUNDLE
    if not root.is_dir():
        root = item['path']  # Native torch_npu and flat unified wheels.
    return {str(path.relative_to(root)): path for path in root.rglob('*')
            if path.is_file() and path.relative_to(root).parts[0] != 'mindspore'
            and not path.relative_to(root).parts[0].endswith(('.dist-info', '.egg-info'))}


def prepare_module(inputs, base):
    """Check module providers and stage their payload under the private package directory."""
    providers = [item for item in inputs if item['name'] == 'torch-npu'
                 or item['entries'].has_option('torch.backends', 'torch_npu')]
    if not providers:
        raise ValueError('At least one input wheel must contain torch_npu or supply --module-wheel')
    provider = providers[0]
    files = module_files(provider)
    if not files:
        raise ValueError('Module wheel has no payload: ' + str(provider['filename']))
    for other in providers[1:]:
        other_files = module_files(other)
        if ({name: digest(path) for name, path in files.items()} !=
                {name: digest(path) for name, path in other_files.items()}):
            raise ValueError('Conflicting torch_npu builds: supply only one build or identical module payloads')
    for name, source in files.items():
        parts = Path(name).parts
        if parts[0].endswith('.data'):
            if parts[1] not in ('purelib', 'platlib'):
                raise ValueError('torch_npu wheel contains files outside its Python packages: ' + name)
            name = str(Path(*parts[2:]))
        destination = base['path'] / BUNDLE / name
        if source == destination:
            continue
        if destination.exists() and digest(destination) != digest(source):
            raise ValueError('Conflicting module file: ' + name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if source.is_relative_to(base['path']):
            source.unlink()  # Migrate a flat unified wheel without leaving shared files.
    for old in ('_mindspore_torch_npu_path.py', 'mindspore_torch_npu.pth', '_mindspore_info.py'):
        (base['path'] / old).unlink(missing_ok=True)
    (base['path'] / '_mindspore_torch_npu_loader.py').write_text(TORCH_NPU_LOADER)
    return provider


def merge_backends(inputs, destination):
    """Keep the original plugin, MPI, GE and largest-dataengine merge behavior."""
    target = destination / 'mindspore'
    for item in inputs[1:]:
        source = item['path'] / 'mindspore'
        plugins = source / 'lib/plugin'
        if plugins.is_dir():
            shutil.copytree(plugins, target / 'lib/plugin', dirs_exist_ok=True)
        for name in BACKEND_LIBRARIES:
            library = source / 'lib' / name
            if library.is_file():
                (target / 'lib').mkdir(parents=True, exist_ok=True)
                shutil.copy2(library, target / 'lib' / name)
        for library in source.glob('_c_dataengine.*.so'):
            existing = list(target.glob('_c_dataengine.*.so'))
            if not existing or library.stat().st_size > max(path.stat().st_size for path in existing):
                for path in existing:
                    path.unlink()
                shutil.copy2(library, target / library.name)
    plugins = target / 'lib/plugin'
    gpu_dirs = [path for path in plugins.glob('gpu*')
                if path.is_dir() and re.fullmatch(r'gpu\d+(?:\.\d+)*', path.name)]
    # Preserve CUDA-versioned libraries; newer versions win if filenames coincide.
    for directory in sorted(gpu_dirs, key=lambda path: Version(path.name[3:])):
        for library in directory.glob('libcuda_ops.so*'):
            common = plugins / 'gpu'
            common.mkdir(exist_ok=True)
            shutil.copy2(library, common / library.name)
            library.unlink()
    (target / 'lib/libakg.so').unlink(missing_ok=True)


def merge_entry_points(entries, incoming):
    """Merge entry points, routing torch_npu hooks through the unified loader."""
    for group in incoming.sections():
        if not entries.has_section(group):
            entries.add_section(group)
        for name, value in incoming[group].items():
            if group == 'torch.backends' and name == 'torch_npu':
                value = '_mindspore_torch_npu_loader:autoload'
            if group == 'console_scripts' and name in ('torch_npu_run', 'mindspore-torch-npu-run'):
                name = 'torch_npu_run'
                value = '_mindspore_torch_npu_loader:run'
            if name in entries[group] and entries[group][name] != value:
                raise ValueError(f'Conflicting entry point: {group}/{name}')
            entries[group][name] = value


def preserve_module_metadata(item, info, metadata):
    """Copy native module metadata and migrate license records when needed."""
    # Keep the module's original metadata and licenses in the single distribution.
    target = info / 'modules' / item['name']
    shutil.copytree(item['info'], target, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('RECORD', 'RECORD.*'))
    for license_file in item['metadata'].get_all('License-File', []):
        if Path(license_file).is_absolute() or '..' in Path(license_file).parts:
            raise ValueError('Invalid license path: ' + license_file)
        license_root = item['info'] / 'licenses' if Version(
            item['metadata']['Metadata-Version']) >= Version('2.4') else item['info']
        destination = info / 'licenses' / item['name'] / license_file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(license_root / license_file, destination)
        if Version(metadata['Metadata-Version']) < Version('2.4'):
            for existing in metadata.get_all('License-File', []):
                target = info / 'licenses' / existing
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(info / existing, target)
        license_name = f"{item['name']}/{license_file}"
        if license_name not in metadata.get_all('License-File', []):
            metadata['License-File'] = license_name
        metadata.replace_header('Metadata-Version', '2.4')


def merge_metadata(base, inputs, requirements):
    """Combine runtime requirements, entry points and licenses from input distributions."""
    info, metadata, entries = base['info'], base['metadata'], base['entries']
    requirements = {str(Requirement(value)) for value in requirements} | set(metadata.get_all('Requires-Dist', []))
    extras = set(metadata.get_all('Provides-Extra', []))
    python_versions = set(metadata.get_all('Requires-Python', []))
    for item in inputs:
        if item is base:
            continue
        native = item['name'] == 'torch-npu'
        extras.update(item['metadata'].get_all('Provides-Extra', []))
        python_versions.update(item['metadata'].get_all('Requires-Python', []))
        for dependency in item['metadata'].get_all('Requires-Dist', []):
            requirement = Requirement(dependency)
            if native and canonicalize_name(requirement.name) == 'torch':
                marker = f'({requirement.marker}) and ' if requirement.marker else ''
                requirement.marker = Marker(marker + 'extra == "torch-npu"')
                extras.add('torch-npu')
            requirements.add(str(requirement))
        merge_entry_points(entries, item['entries'])
        if native:
            preserve_module_metadata(item, info, metadata)
    for header, values in (('Requires-Dist', requirements), ('Provides-Extra', extras),
                           ('Requires-Python', [','.join(sorted(python_versions))] if python_versions else [])):
        del metadata[header]
        for value in sorted(values):
            metadata[header] = value


def write_component_info(package, info, metadata, entries, module_filename):
    """Generate the summary and msinfo data from the versions actually bundled."""
    components = {
        'mindspore': {'name': 'mindspore', 'display_name': 'MindSpore Core',
                      'version': metadata['Version']}}
    for filename in sorted((info / 'modules').glob('*/METADATA')):
        module = BytesParser().parsebytes(filename.read_bytes())
        name = canonicalize_name(module['Name'])
        components[name] = {
            'name': name,
            'display_name': 'Torch Adapter Core (torch_npu)' if name == 'torch-npu' else module['Name'],
            'version': str(Version(module['Version']))}
    if 'torch-npu' not in components:
        # Older unified wheels contain the payload without its original metadata.
        filename = BUNDLE / 'torch_npu/version.py'
        try:
            version = str(Version(read_version(package, filename)))
        except (OSError, ValueError, SyntaxError, TypeError) as error:
            raise ValueError(
                f'{module_filename}: Cannot read bundled torch_npu version '
                f'from {package / filename}: {error}') from error
        components['torch-npu'] = {
            'name': 'torch-npu', 'display_name': 'Torch Adapter Core (torch_npu)',
            'version': version}
    ordered = [components.pop('mindspore'), components.pop('torch-npu')]
    ordered.extend(components[name] for name in sorted(components))
    (info / 'components.json').write_text(
        json.dumps({'components': ordered}, indent=2) + '\n', encoding='utf-8')
    del metadata['Summary']
    metadata['Summary'] = 'MindSpore unified distribution; ' + '; '.join(
        f"{component['display_name']}={component['version']}" for component in ordered)
    if not entries.has_section('console_scripts'):
        entries.add_section('console_scripts')
    entries['console_scripts']['msinfo'] = '_mindspore_torch_npu_loader:msinfo'


def repack(base, inputs, output_dir, requirements, module_filename):
    """Merge standard distribution metadata and let setuptools generate RECORD."""
    package, info, metadata = base['path'], base['info'], base['metadata']
    entries = base['entries']
    if entries.has_option('torch.backends', 'torch_npu'):
        entries['torch.backends']['torch_npu'] = '_mindspore_torch_npu_loader:autoload'
    if entries.has_option('console_scripts', 'mindspore-torch-npu-run'):
        entries.remove_option('console_scripts', 'mindspore-torch-npu-run')
        entries['console_scripts']['torch_npu_run'] = '_mindspore_torch_npu_loader:run'
    if entries.has_option('console_scripts', 'torch_npu_run'):
        entries['console_scripts']['torch_npu_run'] = '_mindspore_torch_npu_loader:run'
    top_levels = set()
    for item in inputs:
        top_level = item['info'] / 'top_level.txt'
        if top_level.exists():
            top_levels.update(top_level.read_text().splitlines())
    merge_metadata(base, inputs, requirements)
    version = str(Version(base['version']))
    metadata.replace_header('Version', version)
    destination_info = package / f'mindspore-{version}.dist-info'
    if info != destination_info:
        info.rename(destination_info)
        info = destination_info
    metadata.replace_header('Name', 'mindspore')
    write_component_info(package, info, metadata, entries, module_filename)
    # Summary is a single logical and physical line, including long local versions.
    (info / 'METADATA').write_bytes(metadata.as_bytes(policy=metadata.policy.clone(max_line_length=0)))
    top_levels.difference_update(path.stem if path.is_file() else path.name
                                 for path in (package / BUNDLE).iterdir())
    top_levels.discard('_mindspore_torch_npu_path')
    top_levels.add('_mindspore_torch_npu_loader')
    top_levels.discard('_mindspore_info')
    if top_levels:
        (info / 'top_level.txt').write_text('\n'.join(sorted(top_levels)) + '\n')
    with (info / 'entry_points.txt').open('w') as stream:
        entries.write(stream)
    wheel_info = BytesParser().parsebytes((info / 'WHEEL').read_bytes())
    del wheel_info['Tag']
    for tag in sorted(base['tags'], key=str):
        wheel_info['Tag'] = str(tag)
    del wheel_info['Root-Is-Purelib']
    wheel_info['Root-Is-Purelib'] = 'false'
    (info / 'WHEEL').write_bytes(wheel_info.as_bytes())
    for old in package.glob('*.egg-info'):
        if old.is_dir():
            shutil.rmtree(old)
        else:
            old.unlink()
    for signature in info.glob('RECORD.*'):
        signature.unlink()
    filename = f"mindspore-{version}-" + base['filename'].name.split('-', 2)[2]
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
        wheel = Path(temporary) / filename
        with WheelFile(wheel, 'w') as archive:
            archive.write_files(str(package))
        destination = output_dir / filename
        wheel.replace(destination)
    destination.with_suffix('.whl.sha256').write_text(f'{digest(destination)}  {filename}\n')
    print(f'Merged wheel: {destination}', flush=True)
    return destination


def module_tags_match(base_tags, module_tags):
    """Check matching Python ABI and architecture for generic Linux and manylinux inputs."""
    if base_tags == module_tags:
        return True
    if len(base_tags) != 1:
        return False
    base = next(iter(base_tags))
    if not base.platform.startswith('linux_'):
        return False
    platform_pattern = r'manylinux(?:[0-9]+|_[0-9]+_[0-9]+)_' + re.escape(base.platform[6:])
    return any(tag.interpreter == base.interpreter and tag.abi == base.abi
               and re.fullmatch(platform_pattern, tag.platform) for tag in module_tags)


def merge(wheels, output_dir, module_wheel=None, requirements=()):
    """Merge compatible MindSpore wheels with a mandatory torch_npu provider."""
    with tempfile.TemporaryDirectory(prefix='merge-wheel-') as temporary:
        work = Path(temporary)
        inputs = [unpack(Path(wheel).resolve(), work / str(index)) for index, wheel in enumerate(wheels)]
        base = inputs[0]
        if base['name'] == 'torch-npu':
            raise ValueError('The first wheel must provide the MindSpore core')
        for item in inputs[1:]:
            if item['tags'] != base['tags'] or Version(item['version']) != Version(base['version']):
                raise ValueError(f"Incompatible MindSpore version or wheel tags: {item['filename']}")
        backends = list(inputs)
        if module_wheel:
            module_wheel = Path(module_wheel).resolve()
            candidates = list(module_wheel.glob('*.whl')) if module_wheel.is_dir() else [module_wheel]
            if len(candidates) != 1:
                raise ValueError('Expected exactly one module wheel: ' + str(candidates))
            module = unpack(candidates[0], work / 'module')
            if module['name'] != 'torch-npu' or not module_tags_match(base['tags'], module['tags']):
                raise ValueError('Incompatible module distribution or wheel tags: ' + str(candidates[0]))
            inputs.append(module)
        provider = prepare_module(inputs, base)
        if len(backends) > 1:
            merge_backends(backends, base['path'])
        return repack(base, inputs, Path(output_dir).resolve(), requirements, provider['filename'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, epilog=
        'The first wheel supplies the MindSpore core; at least one input must contain torch_npu. '
        'Requires setuptools, wheel and packaging; input wheels must have matching tags.')
    parser.add_argument('--require', action='append', default=[],
                        help='Additional runtime requirement established by the build')
    parser.add_argument('--module-wheel', type=Path, help='Native torch_npu wheel or directory containing one wheel')
    parser.add_argument('--output-dir', type=Path, default=Path('output'))
    parser.add_argument('wheels', nargs='+', type=Path)
    args = parser.parse_args()
    merge(args.wheels, args.output_dir, args.module_wheel, args.require)
