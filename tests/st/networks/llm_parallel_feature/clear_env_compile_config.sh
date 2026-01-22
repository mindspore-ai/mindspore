#!/bin/bash
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

python - << EOF

import importlib
import sys
import re
import mindspore


class Modifier:
    def __init__(self, module_name='mindspore.graph._parse.compile_config', package=None):
        self.module_name = module_name
        self.package = package
        self.running = False

    def ModifyAndOverwriteBegin(self):
        if self.running:
            return
        self.running = True
        self.spec = importlib.util.find_spec(self.module_name, self.package)
        self.src = self.spec.loader.get_source(self.module_name)

    def ModifyAndOverwrite(self, key: str, value: str):
        self.src = re.sub(f"{key} = '.*'", f"{key} = '{value}'" , self.src)

    def ModifyAndOverwriteEnd(self):
        fo = open(self.spec.origin, "w")
        fo.write(self.src)
        fo.close()
        self.running = False


m = Modifier()
def Modify(key: str, value: str):
    m.ModifyAndOverwrite(key, value)


##################################################
#          MODIFY COMPILE CONFIG BEGIN
m.ModifyAndOverwriteBegin()
##################################################





Modify('COMPILE_PROFILE', '')
Modify('COMPILE_PROFILE_FINISH_ACTION', '')
Modify('FALLBACK_SUPPORT_LIST_DICT_INPLACE', '')
Modify('FALLBACK_FORCE_ANY', '')
Modify('IF_PARALLEL_CALL', '')
Modify('FOR_HALF_UNROLL', '')
Modify('NOT_WAIT_BRANCH_EVAL', '')
Modify('RECURSIVE_EVAL', '')
Modify('SINGLE_EVAL', '')
Modify('ENABLE_DDE', '')
Modify('DDE_ONLY_MARK', '')
Modify('BOOST_PARSE', '')
Modify('GREED_PARSE', '')
Modify('AMP_ENABLE_ALL_FG', '')
Modify('DUMP_IR_CONFIG', '')
Modify('TRAVERSE_SUBSTITUTIONS_MODE', '')
Modify('PRE_LIFT', '')
Modify('COMPILE_PRINT', '')
Modify('ENABLE_FIX_CODE_LINE', '')
Modify('RECORD_MEMORY', '')
Modify('TRACE_LABEL_WITH_UNIQUE_ID', '')




##################################################
#          MODIFY COMPILE CONFIG END
m.ModifyAndOverwriteEnd()
##################################################
EOF
