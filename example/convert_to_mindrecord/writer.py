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
import os
import pickle
import time
from importlib import import_module
from multiprocessing import Pool

from mindspore.mindrecord import FileWriter


def _exec_task(task_id, parallel_writer=True):
    """
    Execute task with specified task id
    """
    print("exec task {}, parallel: {} ...".format(task_id, parallel_writer))
    imagenet_iter = mindrecord_dict_data(task_id)
    batch_size = 2048
    transform_count = 0
    while True:
        data_list = []
        try:
            for _ in range(batch_size):
                data_list.append(imagenet_iter.__next__())
                transform_count += 1
            writer.write_raw_data(data_list, parallel_writer=parallel_writer)
            print("transformed {} record...".format(transform_count))
        except StopIteration:
            if data_list:
                writer.write_raw_data(data_list, parallel_writer=parallel_writer)
                print("transformed {} record...".format(transform_count))
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Mind record writer')
    parser.add_argument('--mindrecord_script', type=str, default="template",
                        help='path where script is saved')

    parser.add_argument('--mindrecord_file', type=str, default="/tmp/mindrecord",
                        help='written file name prefix')

    parser.add_argument('--mindrecord_partitions', type=int, default=1,
                        help='number of written files')

    parser.add_argument('--mindrecord_workers', type=int, default=8,
                        help='number of parallel workers')

    args = parser.parse_known_args()

    args, other_args = parser.parse_known_args()

    print(args)
    print(other_args)

    with open('mr_argument.pickle', 'wb') as file_handle:
        pickle.dump(other_args, file_handle)

    try:
        mr_api = import_module(args.mindrecord_script + '.mr_api')
    except ModuleNotFoundError:
        raise RuntimeError("Unknown module path: {}".format(args.mindrecord_script + '.mr_api'))

    num_tasks = mr_api.mindrecord_task_number()

    print("Write mindrecord ...")

    mindrecord_dict_data = mr_api.mindrecord_dict_data

    # get number of files
    writer = FileWriter(args.mindrecord_file, args.mindrecord_partitions)

    start_time = time.time()

    # set the header size
    try:
        header_size = mr_api.mindrecord_header_size
        writer.set_header_size(header_size)
    except AttributeError:
        print("Default header size: {}".format(1 << 24))

    # set the page size
    try:
        page_size = mr_api.mindrecord_page_size
        writer.set_page_size(page_size)
    except AttributeError:
        print("Default page size: {}".format(1 << 25))

    # get schema
    try:
        mindrecord_schema = mr_api.mindrecord_schema
    except AttributeError:
        raise RuntimeError("mindrecord_schema is not defined in mr_api.py.")

    # create the schema
    writer.add_schema(mindrecord_schema, "mindrecord_schema")

    # add the index
    try:
        index_fields = mr_api.mindrecord_index_fields
        writer.add_index(index_fields)
    except AttributeError:
        print("Default index fields: all simple fields are indexes.")

    writer.open_and_set_header()

    task_list = list(range(num_tasks))

    # set number of workers
    num_workers = args.mindrecord_workers

    if num_tasks < 1:
        num_tasks = 1

    if num_workers > num_tasks:
        num_workers = num_tasks

    if num_tasks > 1:
        with Pool(num_workers) as p:
            p.map(_exec_task, task_list)
    else:
        _exec_task(0, False)

    ret = writer.commit()

    os.remove("{}".format("mr_argument.pickle"))

    end_time = time.time()
    print("--------------------------------------------")
    print("END. Total time: {}".format(end_time - start_time))
    print("--------------------------------------------")
