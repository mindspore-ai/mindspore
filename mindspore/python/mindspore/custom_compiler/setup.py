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
import re
import subprocess
import shutil
from mindspore import log as logger

OP_HOST = "op_host"
OP_KERNEL = "op_kernel"
SUFFIX_CPP = "cpp"
SUFFIX_H = "h"
BUILD_OUT = "build_out"
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
        script_path = os.path.realpath(__file__)
        dir_path, _ = os.path.split(script_path)
        self.current_path = dir_path
        self.custom_project = os.path.join(dir_path, "CustomProject")
        self.bash_path = shutil.which('bash')
        if not self.bash_path:
            raise RuntimeError('bash not found in PATH')

    @staticmethod
    def check_path(path):
        """check if the path is valid"""
        path_white_list_regex = re.compile(r"[^_A-Za-z0-9/.-]")
        if not isinstance(path, str):
            raise TypeError('Path must be str, not {}.'.format(type(path).__name__))
        if not path:
            raise ValueError("The value of the path cannot be empty.")
        if path_white_list_regex.search(path):  # Check special char
            raise ValueError(
                "Input path contains invalid characters.")
        path = os.path.expanduser(path)  # Consider paths starting with "~"
        if os.path.islink(os.path.abspath(path)):  # when checking link, get rid of the "/" at the path tail if any
            raise ValueError("The value of the path cannot be soft link: {}.".format(path))

        real_path = os.path.realpath(path)

        if len(real_path) > 4096:
            raise ValueError("The length of file path should be less than 4096.")

        if real_path != path and path_white_list_regex.search(real_path):
            raise ValueError(
                "Input path contains invalid characters.")

    def check_args(self):
        """check config"""
        self.check_path(self.args.op_host_path)
        self.check_path(self.args.op_kernel_path)
        if not os.path.isdir(self.args.op_host_path):
            raise ValueError(
                f"Config error! op host path [{self.args.op_host_path}] is not exist,"
                f" please check your set --op_host_path")

        if not os.path.isdir(self.args.op_kernel_path):
            raise ValueError(
                f"Config error! op kernel path [{self.args.op_kernel_path}] is not exist, "
                f"please check your set --op_kernel_path")

        if self.args.soc_version != "":
            support_soc_version = {"ascend310p", "ascend310b", "ascend910", "ascend910b", "ascend910_93"}
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

    def copy_compile_project(self):
        """create compile project by template"""
        template_path = "tools/op_project_templates/ascendc/"
        sample_path = "tools/msopgen/template/operator_demo_projects/ascendc_operator_sample"
        customize_dir = os.path.join(self.args.ascend_cann_package_path, template_path, "customize")
        common_dir = os.path.join(self.args.ascend_cann_package_path, template_path, "common")
        sample_dir = os.path.join(self.args.ascend_cann_package_path, sample_path)

        if os.path.isdir(customize_dir) and os.path.isdir(common_dir):
            shutil.copytree(customize_dir, self.custom_project)
            for item in os.listdir(common_dir):
                src_item_path = os.path.join(common_dir, item)
                dst_item_path = os.path.join(self.custom_project, "/cmake/util/", item)
                if os.path.isfile(src_item_path):
                    shutil.copy2(src_item_path, dst_item_path)
        elif os.path.isdir(sample_dir):
            shutil.copytree(sample_dir, self.custom_project)

        if not os.path.isdir(self.custom_project):
            raise ValueError(
                "Create custom project failed, there is no template {} and sample project {}".format(common_dir,
                                                                                                     sample_dir))

    def generate_compile_project(self):
        """generate compile project"""
        if os.path.exists(self.custom_project) and os.path.isdir(self.custom_project):
            shutil.rmtree(self.custom_project)
        self.copy_compile_project()

        os.chmod(self.custom_project, 0o700)
        for root, dirs, files in os.walk(self.custom_project):
            for d in dirs:
                os.path.join(root, d)
                os.chmod(os.path.join(root, d), 0o700)
            for f in files:
                os.path.join(root, f)
                os.chmod(os.path.join(root, f), 0o700)

        cmake_preset_path = os.path.join(self.custom_project, 'CMakePresets.json')
        with open(cmake_preset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data[CONFIG_KEY_CONFIGUREPRESET][0][CONFIG_KEY_VARIABLE][CONFIG_KEY_COMPUTE_UNIT][
            CONFIG_KEY_VALUE] = "ascend310p;ascend310b;ascend910;ascend910b;ascend910_93"
        with os.fdopen(
                os.open(cmake_preset_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                        0o700), "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        ascend_suffix = {SUFFIX_CPP, SUFFIX_H}
        for item in os.listdir(os.path.join(self.custom_project, OP_HOST)):
            if item.split('.')[-1] in ascend_suffix:
                default_host = os.path.join(self.custom_project, OP_HOST, item)
                os.remove(default_host)

        for item in os.listdir(os.path.join(self.custom_project, OP_KERNEL)):
            if item.split('.')[-1] in ascend_suffix:
                default_kernel = os.path.join(self.custom_project, OP_KERNEL, item)
                os.remove(default_kernel)

    def set_cann_path(self):
        """get cann path by user set or default"""
        if self.args.ascend_cann_package_path != "":
            cann_package_path = self.args.ascend_cann_package_path
        else:
            cann_package_path = os.environ.get('ASCEND_HOME_PATH')
            if cann_package_path is None:
                cann_package_path = "/usr/local/Ascend/cann"

        if not os.path.isdir(cann_package_path):
            logger.error(f"The path '{cann_package_path}' is not a valid path.")

        self.args.ascend_cann_package_path = cann_package_path

    def compile_config(self):
        """create CMakePresets.json by config"""
        with open(os.path.join(self.custom_project, 'CMakePresets.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
        cann_package_path = self.args.ascend_cann_package_path
        logger.info("The ASCEND_CANN_PACKAGE_PATH used for compiling the custom operator is is {}".format(
            cann_package_path))
        data[CONFIG_KEY_CONFIGUREPRESET][0][CONFIG_KEY_VARIABLE][CONFIG_KEY_CANN_PATH][
            CONFIG_KEY_VALUE] = cann_package_path

        if self.args.soc_version != "":
            data[CONFIG_KEY_CONFIGUREPRESET][0][CONFIG_KEY_VARIABLE][CONFIG_KEY_COMPUTE_UNIT][
                CONFIG_KEY_VALUE] = self.args.soc_version

        data[CONFIG_KEY_CONFIGUREPRESET][0][CONFIG_KEY_VARIABLE][CONFIG_KEY_VENDOR_NAME][
            CONFIG_KEY_VALUE] = self.args.vendor_name

        with os.fdopen(
                os.open(os.path.join(self.custom_project, 'CMakePresets.json'), os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                        0o700), "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def clear(self):
        """clear log and build out"""
        if self.args.clear:
            command = ['rm', '-rf', 'build_out', 'install.log', 'build.log', 'generate.log']
            result = subprocess.run(command, shell=False, stderr=subprocess.STDOUT, check=False)
            if result.returncode == 0:
                logger.info("Delete build_out install.log build.log successfully!")
            else:
                logger.error("Delete failed with return code: {} ".format(result.returncode))
                logger.error("Error output:\n{}".format(result.stderr))
                raise RuntimeError("Delete failed!")

    def install_custom(self):
        """install custom run"""
        if self.args.install or self.args.install_path != "":
            logger.info("Install custom opp run in {}".format(self.args.install_path))
            os.environ['ASCEND_CUSTOM_OPP_PATH'] = self.args.install_path
            run_path = []
            build_out_path = os.path.join(self.custom_project, "build_out")
            for item in os.listdir(build_out_path):
                if item.split('.')[-1] == "run":
                    run_path.append(os.path.join(build_out_path, item))
            if not run_path:
                raise RuntimeError("There is no custom run in {}".format(build_out_path))
            result = subprocess.run([self.bash_path, run_path[0]], stdout=os.fdopen(
                os.open("install.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700), "w"),
                                    stderr=subprocess.STDOUT, check=False)
            if result.returncode == 0:
                logger.info("Install custom run opp successfully!")
                logger.info(
                    "Please set [source ASCEND_CUSTOM_OPP_PATH={}/vendors/{}:$ASCEND_CUSTOM_OPP_PATH] to "
                    "make the custom operator effective in the current path.".format(
                        self.args.install_path, self.args.vendor_name))
            else:
                with open('install.log', 'r', encoding='utf-8') as file:
                    for line in file:
                        logger.error(line.strip())
                raise RuntimeError("Install failed!")

    def copy_src(self):
        """copy new src code"""
        ascend_suffix = {SUFFIX_CPP, SUFFIX_H}
        for item in os.listdir(self.args.op_host_path):
            if item.split('.')[-1] in ascend_suffix:
                item_path = os.path.join(self.args.op_host_path, item)
                target_path = os.path.join(self.custom_project, OP_HOST, item)
                if os.path.isfile(item_path):
                    shutil.copy(item_path, target_path)
        for item in os.listdir(self.args.op_kernel_path):
            if item.split('.')[-1] in ascend_suffix:
                item_path = os.path.join(self.args.op_kernel_path, item)
                target_path = os.path.join(self.custom_project, OP_KERNEL, item)
                if os.path.isfile(item_path):
                    shutil.copy(item_path, target_path)

        for root, _, files in os.walk(self.custom_project):
            for f in files:
                _, file_extension = os.path.splitext(f)
                if file_extension == ".sh":
                    os.chmod(os.path.join(root, f), 0o700)

    def compile_custom(self):
        """compile custom op"""
        self.copy_src()
        log_fd = os.open("build.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o700)
        log_file = os.fdopen(log_fd, "w")
        if self.args.ascend_cann_package_path != "":
            result = subprocess.run(
                [self.bash_path, 'start.sh', self.custom_project, self.args.ascend_cann_package_path],
                stdout=log_file,
                stderr=subprocess.STDOUT, check=False)
        else:
            result = subprocess.run([self.bash_path, 'start.sh', self.custom_project],
                                    stdout=log_file,
                                    stderr=subprocess.STDOUT, check=False)

        log_file.close()
        if result.returncode == 0:
            logger.info("Compile custom op successfully!")
        else:
            with open('build.log', 'r', encoding='utf-8') as file:
                for line in file:
                    logger.error(line.strip())
            raise RuntimeError("Compile failed! Please see build.log in current directory for detail info.")

    def compile(self):
        """compile op"""
        self.check_args()
        self.set_cann_path()
        self.generate_compile_project()
        self.compile_config()
        self.compile_custom()
        self.install_custom()
        self.clear()


if __name__ == "__main__":
    config = get_config()
    custom_ooc = CustomOOC(config)
    custom_ooc.compile()
