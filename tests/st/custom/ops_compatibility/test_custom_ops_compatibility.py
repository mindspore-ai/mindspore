# Copyright 2025 Huawei Technologies Co., Ltd
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
# ============================================================================
"""Test whether 'mindspore/ms_custom_ops' can import successfully."""

import os
import sys
from datetime import datetime, timedelta
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from tests.mark_utils import arg_mark


# Repository base URL and platform configuration
REPO_BASE_URL = "https://repo.mindspore.cn/mindspore/ms_custom_ops/version/"
PLATFORM = "ascend"
ARCH = "aarch64"


def extract_text_between(text, start_key, end_key, start_offset=None, end_offset=None):
    """Extract text between two markers from a string.

    Args:
        text: The source text to search in.
        start_key: The starting marker.
        end_key: The ending marker.
        start_offset: Optional offset from the start of start_key.
        end_offset: Optional offset from the end of end_key.

    Returns:
        The extracted text, or None if markers are not found.
    """
    start_idx = text.find(start_key)
    if start_idx < 0:
        return None

    end_idx = text.find(end_key, start_idx)
    if end_idx < 0:
        return None

    slice_start = start_idx + len(start_key) if start_offset is None else start_idx + len(start_key) - start_offset
    slice_end = end_idx if end_offset is None else end_idx + end_offset
    return text[slice_start: slice_end]


def find_main_branch(html_content):
    """Extract the main branch link from HTML content.

    Args:
        html_content: HTML content as a string.

    Returns:
        The main branch name, or None if not found.
    """
    for line in html_content.split('\n'):
        branch = extract_text_between(line, '<a href="main_', '" title="main_', len('main_'))
        if branch is not None:
            return branch
    return None


def find_wheel_file(html_content, py_version):
    """Extract the wheel file name matching the Python version.

    Args:
        html_content: HTML content as a string.
        py_version: Python version string (e.g., 'cp310').

    Returns:
        The wheel file name, or None if not found.
    """
    wheel_suffix = f'-{py_version}-{py_version}-linux_{ARCH}.whl'
    for line in html_content.split('\n'):
        wheel_file = extract_text_between(
            line,
            '<a href="ms_custom_ops-',
            wheel_suffix,
            len('ms_custom_ops-'),
            len(wheel_suffix)
        )
        if wheel_file is not None:
            return wheel_file
    return None


def download_ms_custom_ops_wheel(date_info, install_path):
    """Download the ms_custom_ops wheel for a specific date.

    Args:
        date_info: Date string in YYYYMMDD format.
        install_path: Directory path to save the downloaded wheel.

    Returns:
        Path to the downloaded wheel, or None if download fails.
    """
    os.system('pip install download')
    from download import download

    month_day = date_info[:6]
    branch_link = f'{REPO_BASE_URL}{month_day}/{date_info}'
    branch_file = f'{install_path}/ms_custom_ops_branch.txt'

    print(f"Downloading branch info from: {branch_link}")
    download(branch_link, branch_file, kind="file", replace=True)

    with open(branch_file, 'r') as f:
        main_branch = find_main_branch(f.read())

    if main_branch is None:
        print(f"Main branch not found for date: {date_info}")
        return None

    print(f"Main branch: {main_branch}")

    py_version = f'cp{sys.version_info.major}{sys.version_info.minor}'
    version_link = f'{branch_link}/{main_branch}{PLATFORM}/{ARCH}'
    version_file = f'{install_path}/ms_custom_ops_version.txt'

    print(f"Downloading version info from: {version_link}")
    download(version_link, version_file, kind="file", replace=True)

    with open(version_file, 'r') as f:
        wheel_file = find_wheel_file(f.read(), py_version)

    if wheel_file is None:
        print(f"Wheel file not found for Python version: {py_version}")
        return None

    print(f"Wheel file: {wheel_file}")

    wheel_link = f'{version_link}/{wheel_file}'
    print(f"Downloading wheel from: {wheel_link}")

    path = download(wheel_link, install_path, kind="zip", replace=True)
    print(f"Downloaded to: {path}")

    return path


def download_latest_ms_custom_ops_wheel(install_path):
    """Try to download the latest ms_custom_ops wheel from the past 15 days.

    Args:
        install_path: Directory path to save the downloaded wheel.

    Returns:
        Path to the downloaded wheel, or None if no successful download.
    """
    today = datetime.today()
    for days_back in range(15):
        download_day = today - timedelta(days=days_back)
        formatted_date = download_day.strftime("%Y%m%d")
        print(f"\nTrying date: {formatted_date}")

        path = download_ms_custom_ops_wheel(formatted_date, install_path)
        if path is not None:
            return path

    print("Failed to download ms_custom_ops wheel for the past 15 days")
    return None


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='essential', mem_peak=2.00)
def test_ms_custom_ops_compatibility():
    """Test whether custom ops and pass can import successfully.

    Feature: Test whether custom ops and pass can import successfully.
    Description: Download custom wheel package and try to import it.
    Expectation: Custom wheel package can import successfully.
    """
    local_install_path = '/tmp/local_install'
    print(f'Install path: {local_install_path}')

    os.system(f'rm -rf {local_install_path}')

    wheel_path = download_latest_ms_custom_ops_wheel(local_install_path)
    assert wheel_path is not None, "Failed to download ms_custom_ops wheel"

    sys.path.append(wheel_path)

    import ms_custom_ops
    print(f"ms_custom_ops path: {ms_custom_ops.__path__}")

    expected_path = f'{wheel_path}/ms_custom_ops'
    assert expected_path in ms_custom_ops.__path__, \
        f"Expected path {expected_path} not found in {ms_custom_ops.__path__}"

    os.system(f'rm -rf {local_install_path}')
