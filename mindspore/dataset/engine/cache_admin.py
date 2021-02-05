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
    cache_server = os.path.join(cache_admin_dir, "cache_server")
    os.chmod(cache_admin, stat.S_IRWXU)
    os.chmod(cache_server, stat.S_IRWXU)
    cmd = cache_admin + " " + " ".join(sys.argv[1:])
    sys.exit(subprocess.call(cmd, shell=True))
