# Copyright 2020 Huawei Technologies Co., Ltd
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
"""
@File   : conftest.py
@Desc   : common fixtures for pytest
"""

import pytest
from _pytest.runner import runtestprotocol


def pytest_addoption(parser):
    """
    add runmode option to control running testcase
    """
    parser.addoption(
        "--runmode", action="store", default="nosimu",
        help="simu:simulator backend & nosimu for no backend"
    )
    parser.addoption(
        "--shard", action="store", default="0",
        help="shard id for parallel pipeline"
    )


@pytest.fixture
def test_with_simu(request):
    """
    run PyNative testcases when compiled with simulator
    """
    return request.config.getoption("--runmode") == "simu"


@pytest.fixture
def shard(request):
    """
    specify shard id for parallel pipeline testcases
    """
    return request.config.getoption("--shard")


# https://stackoverflow.com/questions/14121657/how-to-get-test-name-and-test-result-during-run-time-in-pytest
def pytest_runtest_protocol(item, nextitem):
    reports = runtestprotocol(item, nextitem=nextitem)
    for report in reports:
        if report.when == 'call':
            print(f"\n{item.name} --- {report.outcome}")
    return True
