# Copyright 2021 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
This is the entrance to start the cache service
"""

import os
import stat
import subprocess
import sys
import mindspore


def main():
    """Entry point for cache service"""
    cache_admin_dir = os.path.join(os.path.dirname(mindspore.__file__), "bin")
    os.chdir(cache_admin_dir)
    cache_admin = os.path.join(cache_admin_dir, "cache_admin")

    if not os.path.exists(cache_admin):
        raise RuntimeError("Dataset cache is not supported on your mindspore version.")

    cache_server = os.path.join(cache_admin_dir, "cache_server")
    os.chmod(cache_admin, stat.S_IRWXU)
    os.chmod(cache_server, stat.S_IRWXU)

    # set LD_LIBRARY_PATH for libpython*.so
    python_lib_dir = os.path.join(os.path.dirname(mindspore.__file__), "../../..")
    os.environ['LD_LIBRARY_PATH'] = python_lib_dir + ":" + os.environ.get('LD_LIBRARY_PATH')

    # LD_PRELOAD libnnacl.so
    nnacl_lib = os.path.join(os.path.dirname(mindspore.__file__), "lib/libnnacl.so")
    os.environ['LD_PRELOAD'] = nnacl_lib

    sys.exit(subprocess.call([cache_admin] + sys.argv[1:], shell=False, env=os.environ))
