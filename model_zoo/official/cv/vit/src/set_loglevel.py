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
# ============================================================================
"""set_loglevel"""

import os

def set_loglevel(level='info'):
    print('set device global log level to {}'.format(level))
    os.system('/usr/local/Ascend/driver/tools/msnpureport -g {}'.format(level))
    os.system('/usr/local/Ascend/driver/tools/msnpureport -g {} -d 4'.format(level))
    event_log_level = 'enable' if level in ['info', 'debug'] else 'disable'
    print('set device event log level to {}'.format(event_log_level))
    os.system('/usr/local/Ascend/driver/tools/msnpureport -e {}'.format(event_log_level))
    os.system('/usr/local/Ascend/driver/tools/msnpureport -e {} -d 4'.format(event_log_level))
