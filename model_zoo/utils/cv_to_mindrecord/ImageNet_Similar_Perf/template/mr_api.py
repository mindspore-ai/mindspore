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
User-defined API for MindRecord writer.
Two API must be implemented,
  1. mindrecord_task_number()
       # Return number of parallel tasks. return 1 if no parallel
  2. mindrecord_dict_data(task_id)
       # Yield data for one task
       # task_id is 0..N-1, if N is return value of mindrecord_task_number()
"""
import argparse
import pickle

# ## Parse argument

with open('mr_argument.pickle', 'rb') as mindrecord_argument_file_handle:  # Do NOT change this line
    ARG_LIST = pickle.load(mindrecord_argument_file_handle)                # Do NOT change this line
parser = argparse.ArgumentParser(description='Mind record api template')   # Do NOT change this line

# ## Your arguments below
# parser.add_argument(...)

args = parser.parse_args(ARG_LIST)  # Do NOT change this line
print(args)                         # Do NOT change this line


# ## Default mindrecord vars. Comment them unless default value has to be changed.
# mindrecord_index_fields = ['label']
# mindrecord_header_size = 1 << 24
# mindrecord_page_size = 1 << 25


# define global vars here if necessary


# ####### Your code below ##########
mindrecord_schema = {"label": {"type": "int32"}}

def mindrecord_task_number():
    """
    Get task size.

    Return:
       number of tasks
    """
    return 1


def mindrecord_dict_data(task_id):
    """
    Get data dict.

    Yields:
        data (dict): data row which is dict.
    """
    print("task is {}".format(task_id))
    for i in range(256):
        data = {}
        data['label'] = i
        yield data
