# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from __future__ import absolute_import
import os
import platform
import sys
import time
import subprocess
import glob
from copy import deepcopy
from pathlib import Path
from abc import abstractmethod, ABCMeta
from packaging import version
import numpy as np
from mindspore import log as logger
from mindspore._c_expression import MSContext, ms_ctx_param
from ..version import __version__


class EnvChecker(metaclass=ABCMeta):
    """basic class for environment check"""

    @abstractmethod
    def check_env(self):
        pass

    @abstractmethod
    def set_env(self):
        pass

    @abstractmethod
    def check_version(self):
        pass


class CPUEnvChecker(EnvChecker):
    """CPU environment check."""

    def __init__(self, library_path):
        self.library_path = library_path

    def check_env(self):
        pass

    def check_version(self):
        pass

    def set_env(self):
        """set env for cpu"""
        plugin_dir = os.path.dirname(self.library_path)
        akg_dir = os.path.join(plugin_dir, "plugin/cpu")
        if os.getenv('LD_LIBRARY_PATH'):
            os.environ['LD_LIBRARY_PATH'] = akg_dir + ":" + os.environ['LD_LIBRARY_PATH']
        else:
            os.environ['LD_LIBRARY_PATH'] = akg_dir


class GPUEnvChecker(EnvChecker):
    """GPU environment check."""

    def __init__(self, library_path):
        self.version = ["10.1", "11.1", "11.6"]
        self.lib_key_to_lib_name = {'libcu': 'libcuda.so', 'libcudnn': 'libcudnn.so'}
        self.library_path = library_path
        # env
        self.path = os.getenv("PATH")
        self.ld_lib_path = os.getenv("LD_LIBRARY_PATH")

        # check
        self.v = "0"
        self.cuda_lib_path = self._get_lib_path("libcu")
        self.cuda_bin_path = self._get_bin_path("cuda")
        self.cudnn_lib_path = self._get_lib_path("libcudnn")

    def check_env(self):
        pass

    def check_version(self):
        """Check cuda version."""
        version_match = False
        if self._check_version():
            version_match = True
        if not version_match:
            if self.v == "0":
                logger.warning("Can not found cuda libs. Please confirm that the correct "
                               "cuda version has been installed, you can refer to the "
                               "installation guidelines: https://www.mindspore.cn/install")
            else:
                logger.warning(f"MindSpore version {__version__} and cuda version {self.v} does not match, "
                               f"CUDA version [{self.version}] are supported by MindSpore officially. "
                               "Please refer to the installation guide for version matching "
                               "information: https://www.mindspore.cn/install.")
        nvcc_version = self._get_nvcc_version(False)
        if nvcc_version and (nvcc_version not in self.version):
            logger.warning(f"MindSpore version {__version__} and nvcc(cuda bin) version {nvcc_version} "
                           "does not match. Please refer to the installation guide for version matching "
                           "information: https://www.mindspore.cn/install")
        cudnn_version = self._get_cudnn_version()
        if cudnn_version and int(cudnn_version) < 760:
            logger.warning(f"MindSpore version {__version__} and cudDNN version {cudnn_version} "
                           "does not match. Please refer to the installation guide for version matching "
                           "information: https://www.mindspore.cn/install. The recommended version is "
                           "CUDA10.1 with cuDNN7.6.x, CUDA11.1 with cuDNN8.0.x and CUDA11.6 with cuDNN8.5.x.")
        if cudnn_version and int(cudnn_version) < 800 and int(str(self.v).split('.')[0]) > 10:
            logger.warning(f"CUDA version {self.v} and cuDNN version {cudnn_version} "
                           "does not match. Please refer to the installation guide for version matching "
                           "information: https://www.mindspore.cn/install. The recommended version is "
                           "CUDA11.1 with cuDNN8.0.x or CUDA11.6 with cuDNN8.5.x.")

    def get_cudart_version(self):
        """Get cuda runtime version by libcudart.so."""
        for path in self.cuda_lib_path:
            real_path = glob.glob(path + "/lib*/libcudart.so.*.*.*")
            if real_path == []:
                continue
            ls_cudart = subprocess.run(["ls", real_path[0]], timeout=10, text=True,
                                       capture_output=True, check=False)
            if ls_cudart.returncode == 0:
                self.v = ls_cudart.stdout.split('/')[-1].strip('libcudart.so.').strip()
                break
        return self.v

    def set_env(self):
        """set env for gpu"""
        v = self.get_cudart_version()
        v = version.parse(v)
        v_str = str(v.major) + "." + str(v.minor)
        plugin_dir = os.path.dirname(self.library_path)
        akg_dir = os.path.join(plugin_dir, "gpu" + v_str)
        if os.getenv('LD_LIBRARY_PATH'):
            os.environ['LD_LIBRARY_PATH'] = akg_dir + ":" + os.environ['LD_LIBRARY_PATH']
        else:
            os.environ['LD_LIBRARY_PATH'] = akg_dir

    def _get_bin_path(self, bin_name):
        """Get bin path by bin name."""
        if bin_name == "cuda":
            return self._get_cuda_bin_path()
        return []

    def _get_cuda_bin_path(self):
        """Get cuda bin path by lib path."""
        path_list = []
        for path in self.cuda_lib_path:
            path = os.path.abspath(path.strip() + "/bin/")
            if Path(path).is_dir():
                path_list.append(path)
        return np.unique(path_list)

    def _get_nvcc_version(self, is_set_env):
        """Get cuda version by nvcc command."""
        try:
            nvcc_result = subprocess.run(["nvcc", "--version | grep release"],
                                         timeout=3, text=True, capture_output=True, check=False)
        except OSError:
            if not is_set_env:
                for path in self.cuda_bin_path:
                    if Path(path + "/nvcc").is_file():
                        os.environ['PATH'] = path + ":" + os.environ['PATH']
                        return self._get_nvcc_version(True)
            return ""
        result = nvcc_result.stdout
        for line in result.split('\n'):
            if line:
                return line.strip().split("release")[1].split(",")[0].strip()
        return ""

    def _get_cudnn_version(self):
        """Get cudnn version by libcudnn.so."""
        cudnn_version = []
        for path in self.cudnn_lib_path:
            real_path = glob.glob(path + "/lib*/libcudnn.so.*.*")
            if real_path == []:
                continue
            ls_cudnn = subprocess.run(["ls", real_path[0]], timeout=10, text=True,
                                      capture_output=True, check=False)
            if ls_cudnn.returncode == 0:
                cudnn_version = ls_cudnn.stdout.split('/')[-1].strip('libcudnn.so.').strip().split('.')
                if len(cudnn_version) == 2:
                    cudnn_version.append('0')
                break
        version_str = ''.join(cudnn_version)
        return version_str[0:3]

    def _check_version(self):
        """Check cuda version"""
        v = self.get_cudart_version()
        v = version.parse(v)
        v_str = str(v.major) + "." + str(v.minor)
        if v_str not in self.version:
            return False
        return True

    def _get_lib_path(self, lib_name):
        """Get gpu lib path by ldd command."""
        path_list = []
        current_path = os.path.split(os.path.realpath(__file__))[0]
        mindspore_path = os.path.join(current_path, "../lib/plugin")
        try:
            real_path = self.library_path
            if real_path is None or real_path == []:
                logger.error(f"{self.lib_key_to_lib_name[lib_name]} (need by mindspore-gpu) is not found. Please "
                             f"confirm that libmindspore_gpu.so is in directory:{mindspore_path} and the correct cuda "
                             "version has been installed, you can refer to the installation "
                             "guidelines: https://www.mindspore.cn/install")
                return path_list
            ldd_r = subprocess.Popen(['ldd', self.library_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            ldd_result = subprocess.Popen(['/bin/grep', lib_name], stdin=ldd_r.stdout, stdout=subprocess.PIPE)
            result = ldd_result.communicate(timeout=5)[0].decode()
            for i in result.split('\n'):
                path = i.partition("=>")[2]
                if path.lower().find("not found") > 0:
                    logger.error(f"Cuda {self.version} version({lib_name}*.so need by mindspore-gpu) is not found. "
                                 "Please confirm that the path of cuda is set to the env LD_LIBRARY_PATH, or check "
                                 "whether the CUDA version in wheel package and the CUDA runtime in current device "
                                 "matches. Please refer to the installation guidelines: "
                                 "https://www.mindspore.cn/install")
                    continue
                path = path.partition(lib_name)[0]
                if path:
                    path_list.append(os.path.abspath(path.strip() + "../"))
            return np.unique(path_list)
        except subprocess.TimeoutExpired:
            logger.warning("Failed to check cuda version due to the ldd command timeout. Please confirm that "
                           "the correct cuda version has been installed. You can refer to the "
                           "installation guidelines: https://www.mindspore.cn/install")
            return path_list

    def _read_version(self, file_path):
        """Get gpu version info in version.txt."""
        with open(file_path, 'r') as f:
            all_info = f.readlines()
            for line in all_info:
                if line.startswith("CUDA Version"):
                    self.v = line.strip().split("CUDA Version")[1]
                    return self.v
        return self.v


class AscendEnvChecker(EnvChecker):
    """ascend environment check"""

    def __init__(self, library_path):
        self.library_path = library_path
        self.version = ["6.3"]
        atlas_nnae_version = "/usr/local/Ascend/nnae/latest/compiler/version.info"
        atlas_toolkit_version = "/usr/local/Ascend/ascend-toolkit/latest/compiler/version.info"
        hisi_fwk_version = "/usr/local/Ascend/latest/compiler/version.info"
        if os.path.exists(atlas_nnae_version):
            # atlas default path
            self.fwk_path = "/usr/local/Ascend/nnae/latest"
            self.op_impl_path = "/usr/local/Ascend/nnae/latest/opp/built-in/op_impl/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/compiler/ccec_compiler/bin"
            self.fwk_version = atlas_nnae_version
            self.op_path = "/usr/local/Ascend/nnae/latest/opp"
            self.aicpu_path = "/usr/local/Ascend/nnae/latest"
        elif os.path.exists(atlas_toolkit_version):
            # atlas default path
            self.fwk_path = "/usr/local/Ascend/ascend-toolkit/latest"
            self.op_impl_path = "/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/compiler/ccec_compiler/bin"
            self.fwk_version = atlas_toolkit_version
            self.op_path = "/usr/local/Ascend/ascend-toolkit/latest/opp"
            self.aicpu_path = "/usr/local/Ascend/ascend-toolkit/latest"
        elif os.path.exists(hisi_fwk_version):
            # hisi default path
            self.fwk_path = "/usr/local/Ascend/latest"
            self.op_impl_path = "/usr/local/Ascend/latest/opp/built-in/op_impl/ai_core/tbe"
            self.tbe_path = self.fwk_path + "/lib64"
            self.cce_path = self.fwk_path + "/compiler/ccec_compiler/bin"
            self.fwk_version = hisi_fwk_version
            self.op_path = "/usr/local/Ascend/latest/opp"
            self.aicpu_path = "/usr/local/Ascend/latest"
        else:
            # custom or unknown environment
            self.fwk_path = ""
            self.op_impl_path = ""
            self.tbe_path = ""
            self.cce_path = ""
            self.fwk_version = ""
            self.op_path = ""
            self.aicpu_path = ""

        # env
        self.path = os.getenv("PATH")
        self.python_path = os.getenv("PYTHONPATH")
        self.ld_lib_path = os.getenv("LD_LIBRARY_PATH")
        self.ascend_opp_path = os.getenv("ASCEND_OPP_PATH")
        self.ascend_aicpu_path = os.getenv("ASCEND_AICPU_PATH")

        # check content
        self.path_check = "/compiler/ccec_compiler/bin"
        self.python_path_check = "opp/built-in/op_impl/ai_core/tbe"
        self.ld_lib_path_check_fwk = "/lib64"
        self.ld_lib_path_check_addons = "/add-ons"
        self.ascend_opp_path_check = "/op"
        self.v = ""

    def check_env(self):
        self._check_env()

    def check_version(self):
        if not Path(self.fwk_version).is_file():
            logger.warning("Using custom Ascend AI software package (Ascend Data Center Solution) path, package "
                           "version checking is skipped. Please make sure Ascend AI software package (Ascend Data "
                           "Center Solution) version is supported, you can refer to the installation guidelines "
                           "https://www.mindspore.cn/install")
            return

        v = self._read_version(self.fwk_version)
        if v not in self.version:
            v_list = str([x for x in self.version])
            logger.warning(f"MindSpore version {__version__} and Ascend AI software package (Ascend Data Center "
                           f"Solution)version {v} does not match, the version of software package expect one of "
                           f"{v_list}. Please refer to the match info on: https://www.mindspore.cn/install")

    def check_deps_version(self):
        """
            te and hccl wheel package version check
            in order to update the change of 'LD_LIBRARY_PATH' env, run a sub process
        """

        mindspore_version = __version__
        supported_version = self.version
        attention_warning = False
        try:
            from te import version as tever
            v = '.'.join(tever.version.split('.')[0:2])
            if v not in supported_version:
                attention_warning = True
                logger.warning(f"MindSpore version {mindspore_version} and \"te\" wheel package version {v} does not "
                               "match, refer to the match info on: https://www.mindspore.cn/install")
            from hccl import sys_version as hccl_version
            v = '.'.join(hccl_version.__sys_version__.split('.')[0:2])
            if v not in supported_version:
                attention_warning = True
                logger.warning(f"MindSpore version {mindspore_version} and \"hccl\" wheel package version {v} does not "
                               "match, refer to the match info on: https://www.mindspore.cn/install")
        # DO NOT modify exception type to any other, you DO NOT know what kind of exceptions the te will throw.
        # pylint: disable=broad-except
        except Exception as e:
            logger.error("CheckFailed:", e.args)
            logger.error("MindSpore relies on whl packages of \"te\" and \"hccl\" in the \"latest\" "
                         "folder of the Ascend AI software package (Ascend Data Center Solution). Please check whether"
                         " they are installed correctly or not, refer to the match info on: "
                         "https://www.mindspore.cn/install")
        if attention_warning:
            warning_countdown = 3
            for i in range(warning_countdown, 0, -1):
                logger.warning(f"Please pay attention to the above warning, countdown: {i}")
                time.sleep(1)

    def set_env(self):
        plugin_dir = os.path.dirname(self.library_path)
        akg_dir = os.path.join(plugin_dir, "ascend")
        if os.getenv('LD_LIBRARY_PATH'):
            os.environ['LD_LIBRARY_PATH'] = akg_dir + ":" + os.environ['LD_LIBRARY_PATH']
        else:
            os.environ['LD_LIBRARY_PATH'] = akg_dir

        if not self.tbe_path:
            self._check_env()
            return

        try:
            origin_path = deepcopy(sys.path)
            import te  # pylint: disable=unused-import
        # pylint: disable=broad-except
        except Exception:
            sys.path = deepcopy(origin_path)
            if Path(self.tbe_path).is_dir():
                os.environ['LD_LIBRARY_PATH'] = self.tbe_path + ":" + os.environ['LD_LIBRARY_PATH']
            else:
                logger.error(
                    f"No such directory: {self.tbe_path}. Please check if Ascend AI software package (Ascend Data "
                    "Center Solution) is installed correctly.")
                return

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
            logger.error(
                f"No such directory: {self.op_impl_path}. Please check if Ascend AI software package (Ascend Data "
                "Center Solution) is installed correctly.")
            return

        if Path(self.cce_path).is_dir():
            os.environ['PATH'] = self.cce_path + ":" + os.environ['PATH']
        else:
            logger.error(
                f"No such directory: {self.cce_path}. Please check if Ascend AI software package (Ascend Data Center "
                "Solution) is installed correctly.")
            return

        if self.op_path is None:
            pass
        elif Path(self.op_path).is_dir():
            os.environ['ASCEND_OPP_PATH'] = self.op_path
        else:
            logger.error(
                f"No such directory: {self.op_path}. Please check if Ascend AI software package (Ascend Data Center "
                "Solution) is installed correctly.")
            return

        if self.aicpu_path is None:
            pass
        elif Path(self.aicpu_path).is_dir():
            os.environ['ASCEND_AICPU_PATH'] = self.aicpu_path
        else:
            logger.error(
                f"No such directory: {self.aicpu_path}. Please check if Ascend AI software package (Ascend Data Center"
                " Solution) is installed correctly.")
            return

    def _check_env(self):
        """ascend dependence path check"""
        if self.path is None or self.path_check not in self.path:
            logger.warning("Can not find ccec_compiler(need by mindspore-ascend). Please check if you have set env "
                           "PATH, you can refer to the installation guidelines https://www.mindspore.cn/install")

        if self.python_path is None or self.python_path_check not in self.python_path:
            logger.warning(
                "Can not find the tbe operator implementation(need by mindspore-ascend). Please check if you have set "
                "env PYTHONPATH, you can refer to the installation guidelines "
                "https://www.mindspore.cn/install")

        if self.ld_lib_path is None or not (self.ld_lib_path_check_fwk in self.ld_lib_path and
                                            self.ld_lib_path_check_addons in self.ld_lib_path):
            logger.warning("Can not find driver so(need by mindspore-ascend). Please check if you have set env "
                           "LD_LIBRARY_PATH, you can refer to the installation guidelines "
                           "https://www.mindspore.cn/install")

        if self.ascend_opp_path is None or self.ascend_opp_path_check not in self.ascend_opp_path:
            logger.warning(
                "Can not find opp path (need by mindspore-ascend). Please check if you have set env ASCEND_OPP_PATH, "
                "you can refer to the installation guidelines https://www.mindspore.cn/install")

    def _read_version(self, file_path):
        """get ascend version info"""
        with open(file_path, 'r') as f:
            all_info = f.readlines()
            for line in all_info:
                if line.startswith("Version="):
                    full_version = line.strip().split("=")[1]
                    self.v = '.'.join(full_version.split('.')[0:2])
                    return self.v
        return self.v


def check_env(device, _):
    """callback function for checking environment variables"""
    if device.lower() == "ascend":
        env_checker = AscendEnvChecker(None)
        env_checker.check_version()
    elif device.lower() == "gpu":
        env_checker = GPUEnvChecker(None)
    else:
        logger.info(f"Device {device} does not need to check any environment variable, skipping.")
        return
    env_checker.check_env()


def set_env(device, library_path):
    """callback function for setting environment variables"""
    if not os.getenv("MS_DEV_CLOSE_VERSION_CHECK") is None:
        if device in os.environ["MS_DEV_CLOSE_VERSION_CHECK"]:
            return
        os.environ["MS_DEV_CLOSE_VERSION_CHECK"] = os.environ["MS_DEV_CLOSE_VERSION_CHECK"] + ":" + device
    else:
        os.environ["MS_DEV_CLOSE_VERSION_CHECK"] = device

    if device.lower() == "ascend":
        env_checker = AscendEnvChecker(library_path)
    elif device.lower() == "gpu":
        env_checker = GPUEnvChecker(library_path)
    elif device.lower() == "cpu":
        env_checker = CPUEnvChecker(library_path)
    else:
        logger.info(f"Device {device} does not need to check any environment variable, skipping.")
        return

    env_checker.check_version()
    env_checker.set_env()


def check_version_and_env_config():
    """check version and env config"""
    if platform.system().lower() == 'linux':
        # Note: pre-load libgomp.so to solve error like "cannot allocate memory in statis TLS block"
        try:
            import ctypes
            ctypes.cdll.LoadLibrary("libgomp.so.1")
        except OSError:
            logger.warning("Pre-Load Library libgomp.so.1 failed, which might cause TLS memory allocation failure. If "
                           "the failure occurs, you can find a solution in FAQ in "
                           "https://www.mindspore.cn/docs/en/master/faq/installation.html.")
        if not os.getenv("MS_DEV_CLOSE_VERSION_CHECK") is None:
            return
        MSContext.get_instance().register_check_env_callback(check_env)
        MSContext.get_instance().register_set_env_callback(set_env)
        MSContext.get_instance().set_param(ms_ctx_param.device_target,
                                           MSContext.get_instance().get_param(ms_ctx_param.device_target))


def _set_pb_env():
    """Set env variable `PROTOCOL_BUFFERS` to prevent memory overflow."""
    if os.getenv("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") == "cpp":
        logger.info("Current env variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp`. "
                    "When the checkpoint file is too large, "
                    "it may cause memory limit error during load checkpoint file. "
                    "This can be solved by set env `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.")
    elif os.getenv("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") is None:
        logger.info("Setting the env `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` to prevent memory overflow "
                    "during save or load checkpoint file.")
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


def _add_cuda_path():
    """add cuda path on windows."""
    if platform.system().lower() == 'windows':
        cuda_home = os.environ.get('CUDA_PATH')
        if cuda_home is None:
            pass
        else:
            cuda_bin_path = os.path.join(os.environ['CUDA_PATH'], 'bin')
            if sys.version_info >= (3, 8):
                os.add_dll_directory(cuda_bin_path)
            else:
                os.environ['PATH'] += os.pathsep + cuda_bin_path
        cudann_home = os.environ.get('CUDNN_HOME')
        if cudann_home is None:
            pass
        else:
            cuda_home_bin_path = os.path.join(os.environ['CUDNN_HOME'], 'bin')
            if sys.version_info >= (3, 8):
                os.add_dll_directory(cuda_home_bin_path)
            else:
                os.environ['PATH'] += os.pathsep + cuda_home_bin_path


check_version_and_env_config()
_set_pb_env()
_add_cuda_path()
