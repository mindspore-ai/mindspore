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
# ==============================================================================
"""
######################## write mindrecord example ########################
Write mindrecord by data dictionary:
python writer.py --mindrecord_script /YourScriptPath ...
"""
import argparse

import mindspore.dataset as ds

parser = argparse.ArgumentParser(description='Mind record reader')
parser.add_argument('--path', type=str, default="/tmp/cora/mindrecord/cora_mr",
                    help='data file')
args = parser.parse_args()

data_set = ds.MindDataset(args.path)
num_iter = 0
for item in data_set.create_dict_iterator():
    print(item)
    num_iter += 1
print("Total items # is {}".format(num_iter))
