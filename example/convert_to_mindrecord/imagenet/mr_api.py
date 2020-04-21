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
import os
import pickle

######## mindrecord_schema begin ##########
mindrecord_schema = {"label": {"type": "int64"},
                     "data": {"type": "bytes"},
                     "file_name": {"type": "string"}}
######## mindrecord_schema end ##########

######## Frozen code begin ##########
with open('mr_argument.pickle', 'rb') as mindrecord_argument_file_handle:
    ARG_LIST = pickle.load(mindrecord_argument_file_handle)
######## Frozen code end ##########

parser = argparse.ArgumentParser(description='Mind record imagenet example')
parser.add_argument('--label_file', type=str, default="", help='label file')
parser.add_argument('--image_dir', type=str, default="", help='images directory')

######## Frozen code begin ##########
args = parser.parse_args(ARG_LIST)
print(args)
######## Frozen code end ##########


def _user_defined_private_func():
    """
    Internal function for tasks list

    Return:
       tasks list
    """
    if not os.path.exists(args.label_file):
        raise IOError("map file {} not exists".format(args.label_file))

    label_dict = {}
    with open(args.label_file) as file_handle:
        line = file_handle.readline()
        while line:
            labels = line.split(" ")
            label_dict[labels[1]] = labels[0]
            line = file_handle.readline()
    # get all the dir which are n02087046, n02094114, n02109525
    dir_paths = {}
    for item in label_dict:
        real_path = os.path.join(args.image_dir, label_dict[item])
        if not os.path.isdir(real_path):
            print("{} dir is not exist".format(real_path))
            continue
        dir_paths[item] = real_path

    if not dir_paths:
        print("not valid image dir in {}".format(args.image_dir))
        return {}, {}

    dir_list = []
    for label in dir_paths:
        dir_list.append(label)
    return dir_list, dir_paths


dir_list_global, dir_paths_global = _user_defined_private_func()

def mindrecord_task_number():
    """
    Get task size.

    Return:
       number of tasks
    """
    return len(dir_list_global)


def mindrecord_dict_data(task_id):
    """
    Get data dict.

    Yields:
        data (dict): data row which is dict.
    """

    # get the filename, label and image binary as a dict
    label = dir_list_global[task_id]
    for item in os.listdir(dir_paths_global[label]):
        file_name = os.path.join(dir_paths_global[label], item)
        if not item.endswith("JPEG") and not item.endswith(
                "jpg") and not item.endswith("jpeg"):
            print("{} file is not suffix with JPEG/jpg, skip it.".format(file_name))
            continue
        data = {}
        data["file_name"] = str(file_name)
        data["label"] = int(label)

        # get the image data
        image_file = open(file_name, "rb")
        image_bytes = image_file.read()
        image_file.close()
        data["data"] = image_bytes
        yield data
