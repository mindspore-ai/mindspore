# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Resolve an official torch_npu wheel URL without filtering on the host glibc."""
import json
import platform
import re
import subprocess
import sys

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename
from packaging.version import Version

# Pin the Ascend release as well as the torch_npu version for reproducible selection.
RELEASE_TAGS = {'2.13.0': 'v26.2.0-pytorch2.13.0'}


def select_wheel(release, version, interpreter, abi, architecture):
    """Select one release attachment matching the requested version and Python ABI."""
    candidates = set()
    pattern = r'(?:linux|manylinux(?:[0-9]+|_[0-9]+_[0-9]+))_' + re.escape(architecture)
    for item in release['assets']:
        if item.get('type') != 'attach' or not item['name'].endswith('.whl'):
            continue
        name, wheel_version, _, tags = parse_wheel_filename(item['name'])
        if name != 'torch-npu' or wheel_version != Version(version):
            continue
        if any(tag.interpreter == interpreter and tag.abi == abi
               and re.fullmatch(pattern, tag.platform) for tag in tags):
            candidates.add(item['browser_download_url'])
    if len(candidates) != 1:
        raise ValueError(f'Expected one official torch_npu {version} wheel for '
                         f'{interpreter}-{abi}-{architecture}, found {len(candidates)}; '
                         'specify an HTTPS wheel URL explicitly if needed')
    url = candidates.pop()
    if not url.startswith('https://'):
        raise ValueError('Official wheel download must use HTTPS')
    return url


def resolve(version):
    """Use curl's CA configuration to query the pinned official GitCode release."""
    version = str(Version(version))
    if version not in RELEASE_TAGS:
        raise ValueError(f'No GitCode release pinned for {version}; specify an HTTPS wheel URL '
                         'or update RELEASE_TAGS')
    if sys.platform != 'linux':
        raise ValueError('Automatic torch_npu wheel selection requires Linux')
    tag = next(sys_tags())
    url = 'https://api.gitcode.com/api/v5/repos/Ascend/pytorch/releases/tags/' + RELEASE_TAGS[version]
    response = subprocess.run(
        ['curl', '--fail', '--silent', '--show-error', '--location', '--retry', '3',
         '--connect-timeout', '15', '--max-time', '60', '--proto', '=https', '--proto-redir', '=https', url],
        check=True, capture_output=True, text=True)
    release = json.loads(response.stdout)
    return select_wheel(release, version, tag.interpreter, tag.abi, platform.machine())


if __name__ == '__main__':
    try:
        print(resolve(sys.argv[1]))
    except subprocess.CalledProcessError as error:
        sys.exit(f'Cannot query official GitCode release: {error.stderr.strip()}')
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f'Cannot resolve official torch_npu wheel: {error}')
