#!/usr/bin/env bash
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

# Prerequisites: installed unified wheel, matching PyTorch/torchvision, CANN,
# pytest, expecttest and hypothesis; initialized third_party/torch_npu submodule.
# Example: ASCEND_RT_VISIBLE_DEVICES=0 bash tests/torch_npu/run_networks.sh
# PYTHON selects the interpreter; OUTPUT_DIR selects the logs/JUnit directory.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
if [[ $# -ne 0 ]]; then
    echo "Usage: PYTHON=python OUTPUT_DIR=path ASCEND_RT_VISIBLE_DEVICES=0 bash $0" >&2
    exit 2
fi
python_command=$(command -v -- "${PYTHON:-python}") || { echo 'Python executable not found' >&2; exit 1; }
python_command="$(cd "$(dirname "$python_command")" && pwd)/$(basename "$python_command")"
mkdir -p "${OUTPUT_DIR:-$root/build/torch-npu-networks}"
output=$(cd "${OUTPUT_DIR:-$root/build/torch-npu-networks}" && pwd)
rm -f "$output/results.xml"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export TORCH_DEVICE_BACKEND_AUTOLOAD=1 PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
"$python_command" -u - "$root" "$output" <<'PY' 2>&1 | tee "$output/run.log"
"""Run the selected upstream tests against the installed bundled torch_npu."""
import importlib.metadata
from pathlib import Path
import sys

root, output = (Path(argument) for argument in sys.argv[1:])
sys.argv = sys.argv[:1]  # Upstream ResNet also parses process arguments.
test_root = root / 'third_party/torch_npu/test'
cases = [line.strip() for line in (root / 'tests/torch_npu/test_cases.txt').read_text(encoding='utf-8').splitlines()
         if line.strip() and not line.lstrip().startswith('#')]
if not cases or len(cases) != len(set(cases)):
    raise SystemExit('test_cases.txt must contain a nonempty list without duplicate entries')
for case in cases:
    source = (test_root / case.split('::', 1)[0]).resolve()
    if not source.is_relative_to(test_root.resolve()) or not source.is_file():
        raise SystemExit(f'Missing or invalid upstream test: {case}; initialize the torch_npu submodule')

distribution = importlib.metadata.distribution('mindspore')
bundle = Path(distribution.locate_file('mindspore/_bundled/torch_npu/torch_npu')).resolve()
import torch
import torch_npu

loaded = Path(torch_npu.__file__).resolve()
print(f'Python: {sys.executable}')
print(f'MindSpore: {distribution.version}')
print(f'PyTorch: {torch.__version__}')
print(f'torch_npu: {loaded}')
if not bundle.is_dir() or not loaded.is_relative_to(bundle):
    raise SystemExit(f'Expected bundled torch_npu under {bundle}, but loaded {loaded}; use a clean test environment')
print(f'NPU: {torch_npu.npu.get_device_name(0)}')

import pytest


class RequireAllPassed:
    """Reject skipped/xfail tests and incomplete execution in CI."""

    def __init__(self):
        self.passed = 0

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report):
        """Turn upstream skips into failures before the JUnit reporter sees them."""
        if report.skipped or hasattr(report, 'wasxfail'):
            report.outcome = 'failed'
            report.longrepr = f'CI requires this test to pass without skip/xfail: {report.longrepr}'
            if hasattr(report, 'wasxfail'):
                del report.wasxfail
        if report.when == 'call' and report.passed:
            self.passed += 1


results = RequireAllPassed()
exit_code = pytest.main([
    '-v', '-s', '-ra', '--import-mode=importlib', '-o', 'addopts=',
    f'--confcutdir={test_root}', f'--junitxml={output / "results.xml"}',
    *(str(test_root / case) for case in cases),
], plugins=[results])
if results.passed != len(cases):
    print(f'Expected {len(cases)} passing tests, got {results.passed}')
    exit_code = exit_code or 1
raise SystemExit(exit_code)
PY
