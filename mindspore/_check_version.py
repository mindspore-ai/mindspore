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
"""version and config check"""
import os
import sys
import subprocess
from pathlib import Path
from abc import abstractmethod, ABCMeta
from packaging import version
from . import log as logger
from .version import __version__
from .default_config import __package_name__


class EnvChecker(metaclass=ABCMeta):
    """basic class for environment check"""

    @abstractmethod
    def check_env(self, e):
        pass

    @abstractmethod
    def set_env(self):
        pass

    @abstractmethod
    def check_version(self):
        pass


class GPUEnvChecker(EnvChecker):
    """gpu environment check"""

    def __init__(self):
        self.version = ["10.1"]
        self.cuda_path = "/usr/local/cuda"
        if os.path.exists(self.cuda_path):
            # cuda default path
            self.cuda_bin = self.cuda_path + "/bin"
            self.cuda_lib = self.cuda_path + "/lib64"
            self.cuda_version = self.cuda_path + "/version.txt"
        else:
            # custom or unknown environment
            self.cuda_path = ""
            self.cuda_bin = ""
            self.cuda_lib = ""
            self.cuda_version = ""

        # env
        self.path = os.getenv("PATH")
        self.ld_lib_path = os.getenv("LD_LIBRARY_PATH")

        # check
        self.path_check = "/cuda"
        self.ld_lib_path_check = "/cuda"
        self.v = "0"

    def check_env(self, e):
        self._check_env()
        raise e

    def set_env(self):
        if not self.cuda_bin:
            self._check_env()
            return

        if Path(self.cuda_bin).is_dir():
            os.environ['PATH'] = self.cuda_bin + ":" + os.environ['PATH']
        else:
            raise EnvironmentError(
                f"No such directory: {self.cuda_bin}, please check if cuda is installed correctly.")

    def check_version(self):
        if not Path(self.cuda_version).is_file():
            logger.warning("Using custom cuda path, cuda version checking is skiped, please make sure "
                           "cuda version is supported, you can reference to the installation guidelines "
                           "https://www.mindspore.cn/install")
            return

        v = self._read_version(self.cuda_version)
        v = version.parse(v)
        v_str = str(v.major) + "." + str(v.minor)
        if v_str not in self.version:
            logger.warning(f"MindSpore version {__version__} and cuda version {v_str} does not match, "
                           "reference to the match info on: https://www.mindspore.cn/install")

    def _check_env(self):
        """gpu cuda path check"""
        if self.path is None or self.path_check not in self.path:
            logger.warning("Can not find nvcc compiler(need by mindspore-gpu), please check if you have set env "
                           "PATH, you can reference to the installation guidelines https://www.mindspore.cn/install")

        if self.ld_lib_path is None or self.ld_lib_path_check not in self.ld_lib_path:
            logger.warning("Can not find cuda so(need by mindspore-gpu), please check if you have set env "
                           "LD_LIBRARY_PATH, you can reference to the installation guidelines "
                           "https://www.mindspore.cn/install")

    def _read_version(self, file_path):
        """get gpu version info"""
        with open(file_path, 'r') as f:
            all_info = f.readlines()
            for line in all_info:
                if line.startswith("CUDA Version"):
                    self.v = line.strip().split("CUDA Version")[1]
                    return self.v
        return self.v


class AscendEnvChecker(EnvChecker):
    """ascend environment check"""

    def __init__(self):
        self.version = ["1.76.22.3.220"]
        atlas_nnae_version = "/usr/local/Ascend/nnae/latest/fwkacllib/version.info"
        atlas_toolkit_version = "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib/version.info"
        hisi_fwk_version = "/usr/local/Ascend/fwkacllib/version.info"
        if os.path.exists(atlas_nnae_version):
            # atlas default path
            self.fwk_path = "/usr/local/Ascend/nnae/latest/fwkacllib"
            self.op_impl_path = "/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/ccec_compiler/bin"
            self.fwk_version = atlas_nnae_version
            self.op_path = "/usr/local/Ascend/nnae/latest/opp"
        elif os.path.exists(atlas_toolkit_version):
            # atlas default path
            self.fwk_path = "/usr/local/Ascend/ascend-toolkit/latest/fwkacllib"
            self.op_impl_path = "/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/ccec_compiler/bin"
            self.fwk_version = atlas_toolkit_version
            self.op_path = "/usr/local/Ascend/ascend-toolkit/latest/opp"
        elif os.path.exists(hisi_fwk_version):
            # hisi default path
            self.fwk_path = "/usr/local/Ascend/fwkacllib"
            self.op_impl_path = "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/ccec_compiler/bin"
            self.fwk_version = hisi_fwk_version
            self.op_path = "/usr/local/Ascend/opp"
        else:
            # custom or unknown environment
            self.fwk_path = ""
            self.op_impl_path = ""
            self.tbe_path = ""
            self.cce_path = ""
            self.fwk_version = ""
            self.op_path = ""

        # env
        self.path = os.getenv("PATH")
        self.python_path = os.getenv("PYTHONPATH")
        self.ld_lib_path = os.getenv("LD_LIBRARY_PATH")
        self.ascend_opp_path = os.getenv("ASCEND_OPP_PATH")

        # check content
        self.path_check = "/fwkacllib/ccec_compiler/bin"
        self.python_path_check = "opp/op_impl/built-in/ai_core/tbe"
        self.ld_lib_path_check_fwk = "/fwkacllib/lib64"
        self.ld_lib_path_check_addons = "/add-ons"
        self.ascend_opp_path_check = "/op"
        self.v = ""

    def check_env(self, e):
        self._check_env()
        raise e

    def check_version(self):
        if not Path(self.fwk_version).is_file():
            logger.warning("Using custom Ascend 910 AI software package path, package version checking is skiped, "
                           "please make sure Ascend 910 AI software package version is supported, you can reference to "
                           "the installation guidelines https://www.mindspore.cn/install")
            return

        v = self._read_version(self.fwk_version)
        if v not in self.version:
            v_list = str([x for x in self.version])
            logger.warning(f"MindSpore version {__version__} and Ascend 910 AI software package version {v} does not "
                           f"match, the version of software package expect one of {v_list}, "
                           "please reference to the match info on: https://www.mindspore.cn/install")

    def check_deps_version(self):
        """
            te, topi, hccl wheel package version check
            in order to update the change of 'LD_LIBRARY_PATH' env, run a sub process
        """
        input_args = ["--mindspore_version=" + __version__]
        for v in self.version:
            input_args.append("--supported_version=" + v)
        deps_version_checker = os.path.join(os.path.split(os.path.realpath(__file__))[0], "_check_deps_version.py")
        call_cmd = [sys.executable, deps_version_checker] + input_args
        try:
            process = subprocess.run(call_cmd, timeout=3, text=True, capture_output=True, check=False)
            if process.stdout.strip() != "":
                logger.warning(process.stdout.strip())
        except subprocess.TimeoutExpired:
            logger.warning("Package te, topi, hccl version check timed out, skip.")

    def set_env(self):
        if not self.tbe_path:
            self._check_env()
            return

        try:
            # pylint: disable=unused-import
            import te
        # pylint: disable=broad-except
        except Exception:
            if Path(self.tbe_path).is_dir():
                if os.getenv('LD_LIBRARY_PATH'):
                    os.environ['LD_LIBRARY_PATH'] = self.tbe_path + ":" + os.environ['LD_LIBRARY_PATH']
                else:
                    os.environ['LD_LIBRARY_PATH'] = self.tbe_path
            else:
                raise EnvironmentError(
                    f"No such directory: {self.tbe_path}, Please check if Ascend 910 AI software package is "
                    "installed correctly.")

        # check te version after set te env
        self.check_deps_version()

        if Path(self.op_impl_path).is_dir():
            # python path for sub process
            if os.getenv('PYTHONPATH'):
                os.environ['PYTHONPATH'] = self.op_impl_path + ":" + os.environ['PYTHONPATH']
            else:
                os.environ['PYTHONPATH'] = self.op_impl_path
            # sys path for this process
            sys.path.append(self.op_impl_path)

            os.environ['TBE_IMPL_PATH'] = self.op_impl_path
        else:
            raise EnvironmentError(
                f"No such directory: {self.op_impl_path}, Please check if Ascend 910 AI software package is "
                "installed correctly.")

        if Path(self.cce_path).is_dir():
            os.environ['PATH'] = self.cce_path + ":" + os.environ['PATH']
        else:
            raise EnvironmentError(
                f"No such directory: {self.cce_path}, Please check if Ascend 910 AI software package is "
                "installed correctly.")

        if self.op_path is None:
            pass
        elif Path(self.op_path).is_dir():
            os.environ['ASCEND_OPP_PATH'] = self.op_path
        else:
            raise EnvironmentError(
                f"No such directory: {self.op_path}, Please check if Ascend 910 AI software package is "
                "installed correctly.")

    def _check_env(self):
        """ascend dependence path check"""
        if self.path is None or self.path_check not in self.path:
            logger.warning("Can not find ccec_compiler(need by mindspore-ascend), please check if you have set env "
                           "PATH, you can reference to the installation guidelines https://www.mindspore.cn/install")

        if self.python_path is None or self.python_path_check not in self.python_path:
            logger.warning(
                "Can not find tbe op implement(need by mindspore-ascend), please check if you have set env "
                "PYTHONPATH, you can reference to the installation guidelines "
                "https://www.mindspore.cn/install")

        if self.ld_lib_path is None or not (self.ld_lib_path_check_fwk in self.ld_lib_path and
                                            self.ld_lib_path_check_addons in self.ld_lib_path):
            logger.warning("Can not find driver so(need by mindspore-ascend), please check if you have set env "
                           "LD_LIBRARY_PATH, you can reference to the installation guidelines "
                           "https://www.mindspore.cn/install")

        if self.ascend_opp_path is None or self.ascend_opp_path_check not in self.ascend_opp_path:
            logger.warning(
                "Can not find opp path (need by mindspore-ascend), please check if you have set env ASCEND_OPP_PATH, "
                "you can reference to the installation guidelines https://www.mindspore.cn/install")

    def _read_version(self, file_path):
        """get ascend version info"""
        with open(file_path, 'r') as f:
            all_info = f.readlines()
            for line in all_info:
                if line.startswith("Version="):
                    self.v = line.strip().split("=")[1]
                    return self.v
        return self.v

def check_version_and_env_config():
    """check version and env config"""
    if __package_name__.lower() == "mindspore-ascend":
        env_checker = AscendEnvChecker()
    elif __package_name__.lower() == "mindspore-gpu":
        env_checker = GPUEnvChecker()
    else:
        logger.info(f"Package version {__package_name__} does not need to check any environment variable, skipping.")
        return

    try:
        # pylint: disable=unused-import
        from . import _c_expression
        # check version of ascend site or cuda
        env_checker.check_version()

        env_checker.set_env()
    except ImportError as e:
        env_checker.check_env(e)


def _set_pb_env():
    """Set env variable `PROTOCOL_BUFFERS` to prevent memory overflow."""
    if os.getenv("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") == "cpp":
        logger.info("Current env variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp`. "
                    "When the checkpoint file is too large, "
                    "it may cause memory limit error durning load checkpoint file. "
                    "This can be solved by set env `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.")
    elif os.getenv("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") is None:
        logger.info("Setting the env `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` to prevent memory overflow "
                    "during save or load checkpoint file.")
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


check_version_and_env_config()
_set_pb_env()
