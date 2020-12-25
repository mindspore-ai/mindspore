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
"""dependency package version check"""
from argparse import ArgumentParser
import sys

def parse_args():
    """
    parse args .

    Args:

    Returns:
        args.

    Examples:
        >>> parse_args()
    """
    parser = ArgumentParser(description="MindSpore dependency packages version checker.")
    parser.add_argument("--mindspore_version", type=str, help="MindSpore version.")
    parser.add_argument("--supported_version", type=str, action='append', help="Supported environment version.")
    args = parser.parse_args()
    return args

def check_deps_version(mindspore_version, supported_version):
    """
       check te/hccl/topi version

    Args:
        mindspore_version (str): this mindspore package version
        supported_version (str list): supported Ascend 910 AI software package version by this mindspore package

    Returns:
        void
    """
    try:
        from hccl import sys_version as hccl_version
        v = hccl_version.__sys_version__
        if v not in supported_version:
            print(f"MindSpore version {mindspore_version} and \"hccl\" wheel package version {v} does not "
                  "match, reference to the match info on: https://www.mindspore.cn/install")
        import te
        v = te.__version__
        if v not in supported_version:
            print(f"MindSpore version {mindspore_version} and \"te\" wheel package version {v} does not "
                  "match, reference to the match info on: https://www.mindspore.cn/install")
        import topi
        v = topi.__version__
        if v not in supported_version:
            print(f"MindSpore version {mindspore_version} and \"topi\" wheel package version {v} does not "
                  "match, reference to the match info on: https://www.mindspore.cn/install")

    # pylint: disable=broad-except
    except Exception as e:
        print("CheckFailed: ", e.args)
        print("Minspore relies on the 3 whl packages of \"te\", \"topi\" and \"hccl\" in the \"fwkacllib\" "
              "folder of the Ascend 910 AI software package, please check whether they are installed "
              "correctly or not, reference to the match info on: https://www.mindspore.cn/install")

def main():
    args = parse_args()
    check_deps_version(args.mindspore_version, args.supported_version)

if __name__ == "__main__":
    sys.path = sys.path[1:] # avoid the impact of relative path env, only affect this process
    main()
