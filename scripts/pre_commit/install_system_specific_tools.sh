#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

# get os name
uname_s=$(uname -s)
os_name=${uname_s:0:5}

function install_windows_codecheck_tools() {
    # markdownlint
    echo "[INFO] prepare to install markdownlint"
    which gem || (echo "[WARMING] you must install 'ruby' before install 'markdownlint'" && return 1)
    if [ $? -eq "1" ]; then
        return
    fi
    gem sources --add https://gems.ruby-china.com/
    gem install chef-utils -v 16.6.14 && gem install mdl
}

function install_Linux_codecheck_tool() {
    name_release=$(cat </etc/os-release | awk -F'=' '/^NAME/{print $2}')
    if [ "$name_release" == '"Ubuntu"' ] || [ "$name_release" == '"Debian GNU/Linux"' ]; then
        # clang-format
        echo "[INFO] prepare to install clang-format"
        sudo apt-get install clang-format-9
        # check exist rubygems
        echo "[INFO] prepare to install markdownlint"
        gem -v || (sudo apt-get install -y rubygems)
        gem sources --add https://gems.ruby-china.com/
        sudo gem install chef-utils -v 16.6.14 && sudo gem install mdl
        # install shellcheck
        echo "[INFO] prepare to install shellcheck"
        sudo apt-get install shellcheck
    elif [ "$name_release" == '"CentOS Linux"' ] || [ "$name_release" == '"openEuler"' ] || [ "$name_release" == '"Red Hat Enterprise Linux"' ]; then
        echo "[WARMING] This script is calling sudo in order to install tools on your system, please make sure you have sudo privilege and input your password in the following prompt!"
        # clang-format
        echo "[INFO] prepare to install clang-format"
        if [ "$name_release" == '"Red Hat Enterprise Linux"' ] || [ "$name_release" == '"openEuler"' ]; then
            sudo yum install -y git-clang-format.x86_64
        else
            sudo yum install -y centos-release-scl-rh
            sudo yum install -y llvm-toolset-7-git-clang-format
            llvm_path=$(sudo find / -name *clang-format* | grep -E "/clang-format$")
            llvm_home=${llvm_path%/*}
            sudo chmod 666 /etc/profile
            echo "export LLVM_HOME=$llvm_home" >>/etc/profile
            echo "export PATH=\$PATH:\$LLVM_HOME" >>/etc/profile
            sudo chmod 644 /etc/profile
            source /etc/profile
        fi
        # check rubygems exist and version, install markdownlint
        echo "[INFO] prepare to install markdownlint"
        gem -v || (sudo yum install -y rubygems)
        if [ $? -eq "0" ]; then
            gem_version_head=$(gem -v | awk -F'.' '{print $1}')
            gem_version_next=$(gem -v | awk -F'.' '{print $2}')
            if [ "$gem_version_head" -lt 2 ] || [ "$gem_version_head" -eq 2 ] && [ "$gem_version_next" -lt 3 ]; then
                echo "[WARMING] gem version is less then 2.3 to install markdownlint, please upgrade gem"
            else
                gem sources --add https://gems.ruby-china.com/
                sudo gem install chef-utils -v 16.6.14 && sudo gem install mdl
            fi
        fi
        # install shellcheck
        if [ "$name_release" == '"CentOS Linux"' ]; then
            echo "[INFO] prepare to install shellcheck"
            sudo yum install -y epel-release
            sudo yum install -y ShellCheck
        fi
    fi
}

function install_Mac_codecheck_tools() {
    # clang-format
    echo "[INFO] prepare to install clang-format"
    brew install clang-format
    # markdownlint
    echo "[INFO] prepare to install markdownlint"
    brew install ruby
    sudo gem sources --add https://gems.ruby-china.com/
    sudo gem install chef-utils -v 16.6.14 && sudo gem install mdl
    # install shellcheck
    echo "[INFO] prepare to install shellcheck"
    brew install shellcheck
    brew link --overwrite shellcheck
}
if [ "$os_name" == "MINGW" ]; then # Windows
    echo "Windows, git bash"
    install_windows_codecheck_tools
elif [ "$os_name" == "Linux" ]; then # Linux
    echo "GNU/Linux"
    install_Linux_codecheck_tool
elif [ "$os_name" == "Darwi" ]; then # Darwin
    echo "Mac OS X"
    install_Mac_codecheck_tools
else
    echo "unknown os"
    exit 1
fi
# check clang-format version
echo "[INFO] check clang-format version"
if [ "$os_name" == "Linux" ]; then
    name_release=$(cat </etc/os-release | awk -F'=' '/^NAME/{print $2}')
    if [ "$name_release" == '"Ubuntu"' ]; then
        clang-format-9 --version || echo "[WARMING] clang-format not installed!"
    else
        clang-format --version || echo "[WARMING] clang-format not installed!"
    fi
else
    clang-format --version || echo "[WARMING] clang-format not installed!"
fi
# mdl version
echo "[INFO] check markdownlint version"
mdl --version || echo "[WARMING] markdownlint not installed!"
# version of shellcheck
echo "[INFO] check shellcheck version"
shellcheck --version
if [ $? -eq "0" ]; then
    shellcheck_version_first=$(shellcheck -V | awk -F' ' '/^version/{print $2}' | awk -F'.' '{print $1}')
    shellcheck_version_second=$(shellcheck -V | awk -F' ' '/^version/{print $2}' | awk -F'.' '{print $2}')
    shellcheck_version_third=$(shellcheck -V | awk -F' ' '/^version/{print $2}' | awk -F'.' '{print $3}')
    if [ "$shellcheck_version_first" -le 0 ]; then
        if [ "$shellcheck_version_second" -lt 7 ]; then
            echo "[WARMING] shellcheck version less then 0.7.1, please upgrade shellcheck manually"
        elif [ "$shellcheck_version_second" -eq 7 ] && [ "$shellcheck_version_third" -lt 1 ]; then
            echo "[WARMING] shellcheck version less then 0.7.1, please upgrade shellcheck manually"
        fi
    fi
else
    echo "[WARMING] shellcheck not installed!"
fi
