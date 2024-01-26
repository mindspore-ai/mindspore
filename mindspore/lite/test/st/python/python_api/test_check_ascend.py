# Copyright 2024 Huawei Technologies Co., Ltd
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
Test check Ascend.
"""
try:
    from mindspore_lite._check_ascend import AscendEnvChecker
# pylint: disable=broad-except
except Exception as e:
    # For dev, you can():
    # 1. Assume work directory is mindspore root source dir.
    # 2. comment all contents in mindspore/lite/python/api/__init__.py
    # 3. export PYTHONPATH=${PWD}:$PYTHONPATH
    # 4. python -m pytest mindspore/lite/test/st/python/python_api/test_check_ascend.py
    # 5. Note: The last test 'test_check_env' works only when MSLite is installed!
    from mindspore.lite.python.api._check_ascend import AscendEnvChecker # For dev debug only

import tempfile
import logging


def test_read_version():
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as f:
        f.writelines(['Version=7.1.0.5.242',
                      '\nversion_dir=7.0.RC1',
                      '\nrequired_opp_abi_version=">=6.3, <=7.0"'])
        logging.info("wrote version info to tmp file %s", f.name)
        f.flush()

        version = AscendEnvChecker.read_version(f.name)
        assert version == "7.1"


def test_read_version_no_file():
    version = AscendEnvChecker.read_version("/tmp/mslite_file_not_exist")
    assert version is None


def test_read_version_with_bad_contents():
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as f:
        f.writelines(['Version='])
        logging.info("wrote version info to tmp file %s", f.name)
        f.flush()

        version = AscendEnvChecker.read_version(f.name)
        assert version == ''


def test_check_cann_version():
    # pylint: disable=missing-function-docstring
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as f:
        f.writelines(['Version=7.1.0.5.242',
                      '\nversion_dir=7.0.RC1',
                      '\nrequired_opp_abi_version=">=6.3, <=7.0"'])
        logging.info("wrote version info to tmp file %s", f.name)
        f.flush()

        checker = AscendEnvChecker()
        checker.ascend_version_file = f.name
        checker.compatible_cann_versions = ["7.1"]  # mock compatible version
        assert checker.check_cann_version() is True


def test_check_cann_version_bad():
    # pylint: disable=missing-function-docstring
    with tempfile.NamedTemporaryFile(mode='w+', delete=True) as f:
        f.writelines(['Version=7.0.0.5.242',
                      '\nversion_dir=7.0.RC1',
                      '\nrequired_opp_abi_version=">=6.3, <=7.0"'])
        logging.info("wrote version info to tmp file %s", f.name)
        f.flush()

        checker = AscendEnvChecker()
        checker.ascend_version_file = f.name
        checker.compatible_cann_versions = ["7.1"]  # mock compatible version

        assert checker.check_cann_version() is False


def test_check_cann_version_without_version_file():
    checker = AscendEnvChecker()
    checker.ascend_version_file = "/tmp/mslite_file_not_exist"
    checker.compatible_cann_versions = ["7.1"]  # mock compatible version
    assert checker.check_cann_version() is False


def test_check_lib_path():
    checker = AscendEnvChecker()
    checker.env_ld_lib_path = "/usr/local/Ascend/ascend-toolkit/latest/lib64:"  # mock LD_LIBRARY_PATH
    checker.env_ascend_opp_path = "/usr/local/Ascend/ascend-toolkit/latest/opp"  # mock ASCEND_OPP_PATH
    assert checker.check_lib_path()


def test_check_lib_path_good_case_2():
    checker = AscendEnvChecker()
    checker.env_ld_lib_path = "/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64:"  # mock LD_LIBRARY_PATH
    checker.env_ascend_opp_path = "/usr/local/Ascend/ascend-toolkit/latest/opp"  # mock ASCEND_OPP_PATH
    assert checker.check_lib_path()


def test_check_lib_path_good_case_3():
    checker = AscendEnvChecker()
    checker.env_ld_lib_path = "/usr/local/Ascend/ascend-toolkit/latest/opp/lib64:"  # mock LD_LIBRARY_PATH
    checker.env_ascend_opp_path = "/usr/local/Ascend/ascend-toolkit/latest/opp"  # mock ASCEND_OPP_PATH
    assert checker.check_lib_path()


def test_check_lib_path_without_ld_path():
    checker = AscendEnvChecker()
    checker.env_ld_lib_path = None  # mock LD_LIBRARY_PATH
    checker.env_ascend_opp_path = "/usr/local/Ascend/ascend-toolkit/latest/opp"  # mock ASCEND_OPP_PATH
    assert checker.check_lib_path() is False


def test_check_lib_path_with_bad_ld_path():
    checker = AscendEnvChecker()
    checker.env_ld_lib_path = "/usr/local/Ascend/ascend-toolkit/latest/lib"  # mock bad LD_LIBRARY_PATH
    checker.env_ascend_opp_path = "/usr/local/Ascend/ascend-toolkit/latest/opp"  # mock ASCEND_OPP_PATH
    assert checker.check_lib_path() is False


def test_check_lib_path_without_opp_path():
    checker = AscendEnvChecker()
    checker.env_ld_lib_path = "usr/local/Ascend/ascend-toolkit/latest/lib64"  # mock LD_LIBRARY_PATH
    checker.env_ascend_opp_path = None  # mock ASCEND_OPP_PATH
    assert checker.check_lib_path() is False


def test_check_lib_path_with_bad_opp_path():
    checker = AscendEnvChecker()
    checker.env_ld_lib_path = "/usr/local/Ascend/ascend-toolkit/latest/lib64"  # mock LD_LIBRARY_PATH
    checker.env_ascend_opp_path = "/usr/local/Ascend/ascend-toolkit/latest/op"  # mock bad ASCEND_OPP_PATH
    assert checker.check_lib_path() is False


def test_check_env():
    checker = AscendEnvChecker()
    assert checker.check_env()
