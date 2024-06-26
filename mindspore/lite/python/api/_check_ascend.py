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
Check Ascend env and config.
"""
from __future__ import absolute_import
import os
import re
import logging
from abc import ABCMeta
from multiprocessing import Process, Queue


class AscendEnvChecker(metaclass=ABCMeta):
    """Ascend version and env check"""

    def __init__(self):
        # Note: This list contains the compatible Ascend versions for current MSLite version,
        # It MUST be updated when the MSLite Ascend version is upgraded!
        self.compatible_cann_versions = ["7.2", "7.3"]

        self.ascend_home_path = None
        # Get ascend install path in several envs.
        if "ASCEND_HOME_PATH" in os.environ:
            self.ascend_home_path = os.environ["ASCEND_HOME_PATH"]
        if "ASCEND_CUSTOM_PATH" in os.environ:
            self.ascend_home_path = os.path.join(os.environ["ASCEND_CUSTOM_PATH"], "latest")

        if (self.ascend_home_path is None) or (not os.path.isdir(self.ascend_home_path)):
            logging.warning("Found no Ascend home path, please set env ASCEND_HOME_PATH, for example: "
                            "export ASCEND_HOME_PATH=/usr/local/Ascend/latest and source "
                            "${ASCEND_HOME_PATH}/set_env.sh to setup Ascend envs.")
            return

        self.ascend_home_path = os.path.abspath(self.ascend_home_path)
        self.ascend_version_file = os.path.join(self.ascend_home_path, "compiler/version.info")

        self.env_ld_lib_path = os.getenv("LD_LIBRARY_PATH")
        self.env_python_path = os.getenv("PYTHONPATH")
        self.env_ascend_opp_path = os.getenv("ASCEND_OPP_PATH")

    @staticmethod
    def read_version(version_file: str) -> str:
        """
        Read the version number from Ascend version file.

        :param version_file: version file path in Ascend, which contains the line like "Version=7.0.0.1".
        :return: parsed version str, only keep major and minor version number, like '7.0', return None if any error.
        """
        if not os.path.isfile(version_file):
            logging.debug("version file is one valid file path.")
            return None

        with open(version_file, 'r', encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("Version="):
                    full_version = line.strip().split("=")[1]
                    ver = '.'.join(full_version.split('.')[0:2])
                    return ver
        return None

    def check_env(self) -> bool:
        """Check Ascend env"""
        if self.ascend_home_path is None:
            return False
        return self.check_cann_version() and self.check_lib_path() and self.check_python_deps()

    def check_cann_version(self) -> bool:
        v = self.read_version(self.ascend_version_file)
        if v not in self.compatible_cann_versions:
            logging.warning("Ascend AI software version %s does not match any of the expected compatible version "
                            "list %s. Please refer to the match info on: https://www.mindspore.cn/install",
                            v, self.compatible_cann_versions)
            return False
        return True

    def check_lib_path(self) -> bool:
        """Check LI_LIBRARY_PATH is configured correctly."""
        ld_lib_keyword = r"latest/.*lib64"  # the suffix subdirectories of ascend lib path
        ascend_opp_keyword = "latest/opp"  # the suffix subdirectories of ascend opp path

        if self.env_ld_lib_path is None:
            logging.warning("Env LD_LIBRARY_PATH is not set(needed by Mindspore Lite-Ascend). "
                            "Please set the LD_LIBRARY_PATH properly. "
                            "For details, refer to the installation guidelines: https://www.mindspore.cn/install")
            return False
        if re.search(ld_lib_keyword, self.env_ld_lib_path) is None:
            logging.warning("Found no ascend runtime lib path(needed by Mindspore Lite-Ascend). "
                            "Please check whether the env LD_LIBRARY_PATH is set properly, "
                            "for example: LD_LIBRARY_PATH=\"${ASCEND_CUSTOM_PATH}/latest/lib64:${LD_LIBRARY_PATH}\". "
                            "For details, refer to the installation guidelines: https://www.mindspore.cn/install")
            return False

        if self.env_ascend_opp_path is None:
            logging.warning("Env ASCEND_OPP_PATH is not set(needed by Mindspore Lite-Ascend). "
                            "Please set the ASCEND_OPP_PATH properly. "
                            "For details, refer to the installation guidelines: https://www.mindspore.cn/install")
            return False
        if ascend_opp_keyword not in self.env_ascend_opp_path:
            logging.warning("Found no ascend opp path (needed by Mindspore Lite-Ascend). "
                            "Please check whether the env ASCEND_OPP_PATH is set properly. "
                            "for example: ASCEND_OPP_PATH=\"${ASCEND_CUSTOM_PATH}/latest/opp\". "
                            "For details, refer to the installation guidelines: https://www.mindspore.cn/install")
            return False

        return True

    def check_python_path(self) -> bool:
        python_path_keyword = "opp/built-in/op_impl/ai_core/tbe"
        if (self.env_python_path is None) or (python_path_keyword not in self.env_python_path):
            logging.warning(
                "Found no ascend python package path in PYTHONPATH(need by MindSpore Lite-Ascend). "
                "Please make sure the env PYTHONPATH is set properly. "
                "For example: PYTHONPATH=\"${ASCEND_CUSTOM_PATH}/latest/opp/built-in/op_impl/ai_core/tbe\". "
                "For more details, refer to the installation guidelines: https://www.mindspore.cn/install.")
            return False
        return True

    def do_check_python_deps(self, q: Queue):
        """
        Ascend software contains two python package: te, hccl.
        This method checks the version compatibility of these two packages.

        Note: To avoid the conflict with mslite-akg, which imports a different version of te, we MUST launch an isolated
            child process to do the check! The multiprocessing.Queue is adopted to return check result.
        """
        check_result = True
        try:
            check_result = self.check_te_version() and self.check_hccl_version()
        # pylint: disable=broad-except
        except Exception as e:
            # Do NOT modify exception type to any other, you DO NOT know what kind of exceptions the te will throw.
            logging.error("Got exception when checking te/hccl version: %s.", e)
            logging.error("MindSpore Lite relies on Ascend whl packages of \"te\" and \"hccl\" in the \"latest\" "
                          "folder of the Ascend AI software package (Ascend Data Center Solution). Please make sure"
                          " they are installed correctly. For more install guideline, please refer to: "
                          "https://www.mindspore.cn/install")
            q.put(False)
        q.put(check_result)

    def check_te_version(self) -> bool:
        """
        This method may throw exception from module te.
        """

        # pylint: disable=import-outside-toplevel
        from te import version as te_version
        v = '.'.join(te_version.version.split('.')[0:2])
        if v not in self.compatible_cann_versions:
            logging.warning(
                "Ascend \"te\" wheel package version %s does not match any of the compatible Ascend version list "
                "%s. For details, refer to the installation guidelines: https://www.mindspore.cn/install",
                v, self.compatible_cann_versions)
            return False
        return True

    def check_hccl_version(self) -> bool:
        """
        This method may throws exception from module hccl.
        """
        # pylint: disable=unused-import,import-outside-toplevel
        import numpy
        from hccl import sys_version as hccl_version
        v = '.'.join(hccl_version.__sys_version__.split('.')[0:2])
        if v not in self.compatible_cann_versions:
            logging.warning(
                "Ascend \"hccl\" wheel package version %s does not match any of the compatible Ascend version "
                "list %s. For details, refer to the installation guidelines: "
                "https://www.mindspore.cn/install", v, self.compatible_cann_versions)
            return False
        return True

    def check_python_deps(self) -> bool:
        """
        Check PYTHONPATH and necessary python packages.
        """
        if not self.check_python_path():
            return False

        q = Queue()
        p = Process(target=self.do_check_python_deps, args=(q,))
        p.start()
        ret = q.get()  # this will block to wait return value.
        p.join()

        return ret


def ascend_env(func):
    """ascend_env"""
    ascend_checker = AscendEnvChecker()
    ascend_checker.check_env()

    def wrapper():
        func()

    return wrapper


@ascend_env
def check_ascend_env():
    pass
