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
"""setup package for custom compiler tool"""
import argparse
import json
import os
import subprocess
import shutil
from mindspore import log as logger

OP_HOST = "op_host"
OP_KERNEL = "op_kernel"
SUFFIX_CPP = "cpp"
SUFFIX_H = "h"
CONFIG_KEY_CONFIGUREPRESET = "configurePresets"
CONFIG_KEY_VALUE = "value"
CONFIG_KEY_VARIABLE = "cacheVariables"
CONFIG_KEY_CANN_PATH = "ASCEND_CANN_PACKAGE_PATH"
CONFIG_KEY_VENDOR_NAME = "vendor_name"
CONFIG_KEY_COMPUTE_UNIT = "ASCEND_COMPUTE_UNIT"


def get_config():
    """get config from user"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--op_host_path", type=str, required=True)
    parser.add_argument("-k", "--op_kernel_path", type=str, required=True)
    parser.add_argument("--soc_version", type=str, default="")
    parser.add_argument("--ascend_cann_package_path", type=str, default="")
    parser.add_argument("--vendor_name", type=str, default="customize")
    parser.add_argument("--install_path", type=str, default="")
    parser.add_argument("-c", "--clear", action="store_true")
    parser.add_argument("-i", "--install", action="store_true")
    return parser.parse_args()


class CustomOOC():
    """
    Custom Operator Offline Compilation
    """

    def __init__(self, args):
        self.args = args
        self.ori_cmake_preset = ""

    def check_args(self):
        """check config"""
        if not os.path.isdir(self.args.op_host_path):
            raise ValueError(
                f"Config error! op host path [{self.args.op_host_path}] is not exist,"
                f" please check your set --op_host_path")

        if not os.path.isdir(self.args.op_kernel_path):
            raise ValueError(
                f"Config error! op kernel path [{self.args.op_kernel_path}] is not exist, "
                f"please check your set --op_kernel_path")

        if self.args.soc_version != "":
            support_soc_version = {"ascend310p", "ascend310b", "ascend910", "ascend910b", "ascend910c"}
            for item in self.args.soc_version.split(';'):
                if item not in support_soc_version:
                    raise ValueError(
                        f"Config error! Unsupported soc version {self.args.soc_version}! "
                        f"Please check your set --soc_version and use ';' to separate multiple soc_versions, "
                        f"support soc version is {support_soc_version}")

        if self.args.ascend_cann_package_path != "":
            if not os.path.isdir(self.args.ascend_cann_package_path):
                raise ValueError(
                    f"Config error! ascend cann package path [{self.args.ascend_cann_package_path}] is not valid path, "
                    f"please check your set --ascend_cann_package_path")

        if self.args.install or self.args.install_path != "":
            if self.args.install_path == "":
                opp_path = os.environ.get('ASCEND_OPP_PATH')
                if opp_path is None:
                    raise ValueError(
                        "Config error! Can not find install path, please set install path by --install_path")
                self.args.install_path = opp_path

            if not os.path.isdir(self.args.install_path):
                raise ValueError(
                    f"Install path [{self.args.install_path}] is not valid path, please check your set"
                    f" --install_path is set correctly")

    def compile_config(self):
        """create CMakePresets.json by config"""
        script_path = os.path.abspath(__file__)
        dir_path, _ = os.path.split(script_path)
        with open(os.path.join(dir_path, 'CMakePresetsDefault.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.ori_cmake_preset = data
        if self.args.ascend_cann_package_path != "":
            cann_package_path = self.args.ascend_cann_package_path
        else:
            cann_package_path = os.environ.get('ASCEND_AICPU_PATH')
            if cann_package_path is None:
                raise ValueError("Config error!  Can not find cann package path, "
                                 "please set cann package path by --ascend_cann_package_path.")
        if not os.path.isdir(cann_package_path):
            logger.error(f"The path '{cann_package_path}' is not a valid path.")
        logger.info("ASCEND_CANN_PACKAGE_PATH is {}".format(cann_package_path))
        data[CONFIG_KEY_CONFIGUREPRESET][0][CONFIG_KEY_VARIABLE][CONFIG_KEY_CANN_PATH][
            CONFIG_KEY_VALUE] = cann_package_path

        if self.args.soc_version != "":
            data[CONFIG_KEY_CONFIGUREPRESET][0][CONFIG_KEY_VARIABLE][CONFIG_KEY_COMPUTE_UNIT][
                CONFIG_KEY_VALUE] = self.args.soc_version

        data[CONFIG_KEY_CONFIGUREPRESET][0][CONFIG_KEY_VARIABLE][CONFIG_KEY_VENDOR_NAME][
            CONFIG_KEY_VALUE] = self.args.vendor_name

        with os.fdopen(
                os.open(os.path.join(dir_path, 'CMakePresets.json'), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700),
                "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def clear(self):
        """clear log and build out"""
        if self.args.clear:
            command = ['rm', '-rf', 'build_out', 'install.log', 'build.log']
            result = subprocess.run(command, shell=False, stderr=subprocess.STDOUT)
            if result.returncode == 0:
                logger.info("Delete build_out install.log build.log successfully!")
            else:
                logger.error("Delete failed with return code: {} ".format(result.returncode))
                logger.error("Error output:\n{}".format(result.stderr))
                raise RuntimeError("Delete failed!")

    def install_custom(self):
        """install custom run"""
        if self.args.install or self.args.install_path != "":
            script_path = os.path.abspath(__file__)
            project_path = os.path.dirname(script_path)
            logger.info("Install custom opp run in {}".format(self.args.install_path))
            os.environ['ASCEND_CUSTOM_OPP_PATH'] = self.args.install_path
            result = subprocess.run(['bash', project_path + '/build_out/*.run'], stdout=os.fdopen(
                os.open("install.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700), "w"),
                                    stderr=subprocess.STDOUT)
            if result.returncode == 0:
                logger.info("Install custom run opp successfully!")
                logger.info(
                    "Please set [source ASCEND_CUSTOM_OPP_PATH={}/vendors/{}:$ASCEND_CUSTOM_OPP_PATH] to "
                    "make the custom operator effective in the current path.".format(
                        self.args.install_path, self.args.vendor_name))
            else:
                with open('install.log', 'r') as file:
                    for line in file:
                        logger.error(line.strip())
                raise RuntimeError("Install failed!")

    def clean_and_copy_src(self):
        """clean cache and copy new src code"""
        script_path = os.path.abspath(__file__)
        project_path = os.path.dirname(script_path)
        ascend_suffix = {SUFFIX_CPP, SUFFIX_H}
        for item in os.listdir(os.path.join(project_path, OP_HOST)):
            if item.split('.')[-1] in ascend_suffix:
                os.remove(os.path.join(project_path, OP_HOST, item))

        for item in os.listdir(os.path.join(project_path, OP_KERNEL)):
            if item.split('.')[-1] in ascend_suffix:
                os.remove(os.path.join(project_path, OP_KERNEL, item))

        for item in os.listdir(self.args.op_host_path):
            if item.split('.')[-1] in ascend_suffix:
                item_path = os.path.join(self.args.op_host_path, item)
                target_path = os.path.join(project_path, OP_HOST, item)
                if os.path.isfile(item_path):
                    shutil.copy(item_path, target_path)
        for item in os.listdir(self.args.op_kernel_path):
            if item.split('.')[-1] in ascend_suffix:
                item_path = os.path.join(self.args.op_kernel_path, item)
                target_path = os.path.join(project_path, OP_KERNEL, item)
                if os.path.isfile(item_path):
                    shutil.copy(item_path, target_path)

        for root, _, files in os.walk(project_path):
            for f in files:
                _, file_extension = os.path.splitext(f)
                if file_extension == ".sh":
                    os.chmod(os.path.join(root, f), 0o700)

    def compile_custom(self):
        """compile custom op"""
        self.clean_and_copy_src()
        log_fd = os.open("build.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700)
        log_file = os.fdopen(log_fd, "w")
        script_path = os.path.abspath(__file__)
        project_path = os.path.dirname(script_path)
        build_path = os.path.join(project_path, 'build.sh')
        bash_command = 'unset ASCEND_CUSTOM_OPP_PATH && ' + build_path
        result = subprocess.run(['bash', '-c', bash_command],
                                stdout=log_file,
                                stderr=subprocess.STDOUT)
        log_file.close()
        if result.returncode == 0:
            logger.info("Compile custom op successfully!")
        else:
            with open('build.log', 'r') as file:
                for line in file:
                    logger.debug(line.strip())
            raise RuntimeError("Compile failed! Please see build.log in current directory for detail info.")

    def compile(self):
        """compile op"""
        self.check_args()
        self.compile_config()
        self.compile_custom()
        self.install_custom()
        self.clear()


if __name__ == "__main__":
    config = get_config()
    custom_ooc = CustomOOC(config)
    custom_ooc.compile()
