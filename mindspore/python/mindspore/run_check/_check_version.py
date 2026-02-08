# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
from pathlib import Path
from abc import abstractmethod, ABCMeta
from packaging import version
import numpy as np
from mindspore import log as logger
from mindspore.log import vlog_print
from ..version import __version__


class EnvChecker(metaclass=ABCMeta):
    """basic class for environment check"""

    @abstractmethod
    def check_env(self):
        """check dependency"""

    @abstractmethod
    def set_env(self):
        pass

    @abstractmethod
    def check_version(self):
        pass

    @staticmethod
    def _concat_variable(env_name, env_value):
        """concat value to the beginning of env specified by env_name"""
        if not os.getenv(env_name, ""):
            os.environ[env_name] = env_value
        else:
            paths = os.environ[env_name].split(':')
            if paths and paths[0] == env_value:
                return
            if env_value not in paths:
                os.environ[env_name] = env_value + ':' + os.environ[env_name]
            else:
                # move env_value to beginning
                new_paths = [p for p in paths if p != env_value]
                new_paths.insert(0, env_value)
                os.environ[env_name] = ':'.join(new_paths)


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
        EnvChecker._concat_variable('LD_LIBRARY_PATH', akg_dir)


class GPUEnvChecker(EnvChecker):
    """GPU environment check."""

    def __init__(self, library_path):
        self.version = ["10.1", "11.1", "11.6"]
        self.lib_key_to_lib_name = {'libcudart': 'libcuda.so', 'libcudnn': 'libcudnn.so'}
        self.library_path = library_path
        # env
        self.path = os.getenv("PATH")
        self.ld_lib_path = os.getenv("LD_LIBRARY_PATH")

        # check
        self.v = "0"
        self.cuda_lib_path = self._get_lib_path("libcudart")
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
                logger.warning("Cannot find cuda libs. Please confirm that the correct "
                               "cuda version has been installed. Refer to the "
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
        if cudnn_version and int(cudnn_version) < 800 and int(str(self.v).split('.', maxsplit=1)[0]) > 10:
            logger.warning(f"CUDA version {self.v} and cuDNN version {cudnn_version} "
                           "does not match. Please refer to the installation guide for version matching "
                           "information: https://www.mindspore.cn/install. The recommended version is "
                           "CUDA11.1 with cuDNN8.0.x or CUDA11.6 with cuDNN8.5.x.")

    def get_cudart_version(self):
        """Get cuda runtime version by libcudart.so."""
        for path in self.cuda_lib_path:
            real_path = glob.glob(path + "/lib*/libcudart.so.*.*.*")
            # /usr/lib/x86_64-linux-gnu is a default dir for cuda10.1 on ubuntu.
            if not real_path:
                real_path = glob.glob(path + "/x86_64-linux-gnu/libcudart.so.*.*.*")
            if not real_path:
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
        EnvChecker._concat_variable('LD_LIBRARY_PATH', akg_dir)
        os.environ['CUDA_CACHE_MAXSIZE'] = "4000000000"

    def _get_bin_path(self, bin_name):
        """Get bin path by bin name."""
        if bin_name == "cuda":
            return self._get_cuda_bin_path()
        return []

    def _get_cuda_bin_path(self):
        """Get cuda bin path by lib path."""
        path_list = []
        for path in self.cuda_lib_path:
            path = os.path.realpath(path.strip() + "/bin/")
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
            if not real_path:
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
            with subprocess.Popen(['ldd', self.library_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE) as ldd_r:
                with subprocess.Popen(['/bin/grep', lib_name], stdin=ldd_r.stdout, stdout=subprocess.PIPE) as ldd_res:
                    result = ldd_res.communicate(timeout=5)[0].decode()
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
                    path_list.append(os.path.realpath(path.strip() + "../"))
            return np.unique(path_list)
        except subprocess.TimeoutExpired:
            logger.warning("Failed to check cuda version due to the ldd command timeout. Please confirm that "
                           "the correct cuda version has been installed. For details, refer to the "
                           "installation guidelines: https://www.mindspore.cn/install")
            return path_list

    def _read_version(self, file_path):
        """Get gpu version info in version.txt."""
        with open(file_path, 'r', encoding='utf-8') as f:
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
        self.version = ["8.2", "8.3", "8.5"]

        # env
        self.path = os.getenv("PATH")
        self.python_path = os.getenv("PYTHONPATH")
        self.ld_lib_path = os.getenv("LD_LIBRARY_PATH")
        self.ascend_opp_path = os.getenv("ASCEND_OPP_PATH")
        self.ascend_home_path = os.getenv("ASCEND_HOME_PATH")
        if self.ascend_home_path is not None:
            self.compiler_version = self.ascend_home_path + "/compiler/version.info"
        else:
            self.compiler_version = ""
        # check content
        self.python_path_check = "opp/built-in/op_impl/ai_core/tbe"
        self.ld_lib_path_check_fwk = "/lib64"
        self.ascend_opp_path_check = "/opp"
        self.v = ""

    def check_custom_version(self):
        """custom op version check"""

        if not Path(self.compiler_version).is_file():
            return True

        cur_version = self._read_version(self.compiler_version)
        custom_version_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           "../lib/plugin/ascend/custom_ascendc_910b/version.info")
        with open(custom_version_path, 'r', encoding='utf-8') as f:
            all_info = f.readlines()
            for line in all_info:
                full_version = line.strip().split("=")[1]
                compile_version = '.'.join(full_version.split('.')[0:2])
        if cur_version == compile_version:
            return True

        logger.warning(
            f"The version {compile_version} used for compiling the custom operator does not match "
            f"Ascend AI software package version {cur_version} in the current environment.")
        return False

    def check_env(self):
        self._check_env()

    def check_version(self):
        if not Path(self.compiler_version).is_file():
            logger.warning("Using custom Ascend AI software package (Ascend Data Center Solution) path, package "
                           "version checking is skipped. Please make sure Ascend AI software package (Ascend Data "
                           "Center Solution) version is supported. For details, refer to the installation guidelines "
                           "https://www.mindspore.cn/install")
            return

        v = self._read_version(self.compiler_version)
        if v not in self.version:
            v_list = str(list(self.version))
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
                               "match. For details, refer to the installation guidelines: "
                               "https://www.mindspore.cn/install")
        # DO NOT modify exception type to any other, you DO NOT know what kind of exceptions the te will throw.
        # pylint: disable=broad-except
        except Exception as e:
            logger.error(f"CheckFailed: {e}")
            logger.critical("MindSpore relies on whl package \"te\" in "
                            "Ascend AI software package (Ascend Data Center Solution). Please check whether they are "
                            "installed correctly or not, refer to the match info on: https://www.mindspore.cn/install")
        if attention_warning:
            warning_countdown = 3
            for i in range(warning_countdown, 0, -1):
                logger.warning(f"Please pay attention to the above warning, countdown: {i}")
                time.sleep(1)

    def set_env(self):
        curr_path = os.path.realpath(os.path.dirname(__file__))
        cust_aicpu_path = os.path.realpath(os.path.join(curr_path, "../lib/plugin/ascend/custom_aicpu_ops"))
        cust_aicore_path = os.path.realpath(os.path.join(curr_path, "../lib/plugin/ascend/custom_aicore_ops"))
        cust_ascendc_ascend910b_path = os.path.realpath(
            os.path.join(curr_path, "../lib/plugin/ascend/custom_ascendc_910b"))
        if os.getenv('ASCEND_CUSTOM_OPP_PATH'):
            os.environ['ASCEND_CUSTOM_OPP_PATH'] = os.environ['ASCEND_CUSTOM_OPP_PATH'] + ":" + \
                                                   cust_ascendc_ascend910b_path + ":" + cust_aicore_path + ":" + \
                                                   cust_aicpu_path
        else:
            os.environ['ASCEND_CUSTOM_OPP_PATH'] = cust_ascendc_ascend910b_path + ":" + cust_aicore_path + ":" + \
                                                   cust_aicpu_path
        # Ignore ge infer missing error. To be removed after infers are completed.
        os.environ['FAST_IGNORE_INFER_ERROR'] = "1"
        os.environ['IGNORE_INFER_ERROR'] = "1"
        plugin_dir = os.path.dirname(self.library_path)
        akg_dir = os.path.join(plugin_dir, "ascend")
        EnvChecker._concat_variable('LD_LIBRARY_PATH', akg_dir)

        self._check_env()

        # check te version after set te env
        self.check_deps_version()

    def _check_env(self):
        """ascend dependence path check"""
        if self.python_path is None or self.python_path_check not in self.python_path:
            logger.warning(
                "Cannot find the tbe operator implementation(need by mindspore-ascend). Please check whether the "
                "Environment Variable PYTHONPATH is set. For details, refer to the installation guidelines: "
                "https://www.mindspore.cn/install")

        if self.ld_lib_path is None or self.ld_lib_path_check_fwk not in self.ld_lib_path:
            logger.warning("Cannot find driver so(need by mindspore-ascend). Please check whether the "
                           "Environment Variable LD_LIBRARY_PATH is set. For details, refer to the installation "
                           "guidelines: https://www.mindspore.cn/install")

        if self.ascend_opp_path is None or self.ascend_opp_path_check not in self.ascend_opp_path:
            logger.warning(
                "Cannot find opp path (need by mindspore-ascend). Please check whether the Environment Variable "
                "ASCEND_OPP_PATH is set. For details, refer to the installation guidelines: "
                "https://www.mindspore.cn/install")

    def _read_version(self, file_path):
        """get ascend version info"""
        with open(file_path, 'r', encoding='utf-8') as f:
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
                           "the failure occurs, please refer to the FAQ for a solution: "
                           "https://www.mindspore.cn/docs/en/master/faq/installation.html.")
        from mindspore._c_expression import MSContext, ms_ctx_param
        MSContext.get_instance().register_check_env_callback(check_env)
        MSContext.get_instance().register_set_env_callback(set_env)
        MSContext.get_instance().set_device_target_inner(MSContext.get_instance().get_param(ms_ctx_param.device_target))


def _set_pb_env():
    """Set env variable `PROTOCOL_BUFFERS` to prevent memory overflow."""
    if os.getenv("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") == "cpp":
        logger.info("Current env variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp`. "
                    "When the checkpoint file is too large, "
                    "it may cause memory limit error during load checkpoint file. "
                    "This can be solved by setting env `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.")
    elif os.getenv("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") is None:
        logger.info("Setting the env `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` to prevent memory overflow "
                    "during save or load checkpoint file.")
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


def _check_dir_path_safety(dir_path):
    """Check safety of env directory path."""
    if not os.path.exists(dir_path):
        logger.warning(f"Path {dir_path} not exists.")
        return False

    if not os.path.isdir(dir_path):
        logger.warning(f"Path {dir_path} is not a directory.")
        return False

    # check if path is suspicious
    suspicious_patterns = [
        "/tmp/", "/var/tmp/", "/dev/", "/proc/",
        "\\temp\\", "\\windows\\temp\\",
        "appdata", "local\\temp"
    ]
    lower_path = dir_path.lower()
    for pattern in suspicious_patterns:
        if pattern in lower_path:
            logger.warning(f"Path {dir_path} is suspicious.")
            return False

    # check whether the path points to a system-critical directory
    critical_dirs = [
        "/bin", "/sbin", "/usr/bin", "/usr/sbin",
        "/windows", "/system32", "c:\\windows"
    ]
    for critical_dir in critical_dirs:
        if critical_dir in lower_path:
            logger.warning(f"Path {dir_path} points to a system-critical directory.")
            return False

    return True


def check_cuda_path_safety(cuda_path):
    """Check safety of cuda path."""
    if not _check_dir_path_safety(cuda_path):
        return False

    expected_files = ["nvcc", "cudart.dll", "cudart.so"]
    has_expected_content = False
    for expected_file in expected_files:
        if os.path.exists(os.path.join(cuda_path, "bin", expected_file)):
            has_expected_content = True
            break

    if not has_expected_content:
        logger.warning(f"The directory {cuda_path} does not contain the typical file structure of CUDA")
        return False

    return True


def check_cudnn_path_safety(cudnn_path):
    """Check safety of cudnn path."""
    if not _check_dir_path_safety(cudnn_path):
        return False

    expected_files = [
        "include/cudnn.h",
        "lib64/libcudnn.so",  # Linux
        "lib/libcudnn.dylib",  # macOS
        "lib/x64/cudnn.lib",  # Windows
        "bin/cudnn64_7.dll"   # Windows
    ]
    found_files = []
    for expected_file in expected_files:
        full_path = os.path.join(cudnn_path, expected_file)
        if os.path.exists(full_path):
            found_files.append(expected_file)

    if not found_files:
        logger.warning(f"The directory {cudnn_path} does not contain the typical file structure of CUDNN")
        return False

    return True


def _add_cuda_path():
    """add cuda path on windows."""
    if platform.system().lower() == 'windows':
        cuda_home = os.environ.get('CUDA_PATH')
        if cuda_home is None:
            pass
        else:
            if not check_cuda_path_safety(cuda_home):
                logger.error(f"CUDA_PATH {cuda_home} is not safe, skip add cuda path.")
                return
            cuda_bin_path = os.path.join(os.environ['CUDA_PATH'], 'bin')
            if sys.version_info >= (3, 8):
                os.add_dll_directory(cuda_bin_path)
            else:
                os.environ['PATH'] += os.pathsep + cuda_bin_path
        cudnn_home = os.environ.get('CUDNN_HOME')
        if cudnn_home is None:
            pass
        else:
            if not check_cudnn_path_safety(cudnn_home):
                logger.error(f"CUDNN_HOME {cuda_home} is not safe, skip add cudnn home.")
                return
            cuda_home_bin_path = os.path.join(os.environ['CUDNN_HOME'], 'bin')
            if sys.version_info >= (3, 8):
                os.add_dll_directory(cuda_home_bin_path)
            else:
                os.environ['PATH'] += os.pathsep + cuda_home_bin_path


_set_pb_env()
check_version_and_env_config()
_add_cuda_path()
vlog_print("1", "ME", __file__, sys._getframe().f_lineno, "Initialization MindSpore.")
