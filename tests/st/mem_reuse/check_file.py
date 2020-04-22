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
import os
import filecmp

curr_path = os.path.abspath(os.curdir)
file_memreuse = curr_path + "/mem_reuse_check/memreuse.ir"
file_normal = curr_path + "/mem_reuse_check/normal_mem.ir"
checker = os.path.exists(file_memreuse)
assert (checker, True)
checker = os.path.exists(file_normal)
assert (checker, True)
checker = filecmp.cmp(file_memreuse, file_normal)
assert (checker, True)
