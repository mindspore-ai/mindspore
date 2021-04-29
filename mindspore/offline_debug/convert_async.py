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
"""Module to provide conversion capabalities from .timestamp async dump files to .npy."""
import site
import os
DIR_PATH = "/usr/local/Ascend/toolkit/tools/operator_cmp/compare/"
if not os.path.exists(DIR_PATH):
    raise ValueError("Directory " + DIR_PATH + " does not exist. Please install Ascend toolkit.")
site.addsitedir(DIR_PATH)
#pylint: disable=wrong-import-position
import argparse
import csv
from dump_data_parser import DumpDataParser
from shape_conversion import FormatConversionMain
import utils
#pylint: enable=wrong-import-position


def handle_multi_process(convert_obj, files):
    """Convert async format files to npy in a multithreaded manner"""
    #pylint: disable=W0212
    return_code = utils.VECTOR_COMPARISON_NONE_ERROR
    convert_obj.progress = utils.Progress(len(files))
    multi_process_file_list = []
    big_file_list = []
    max_file_size = convert_obj._get_max_file_size()
    for cur_file in files:
        cur_path = cur_file
        if os.path.isfile(cur_path):
            if os.path.getsize(cur_path) > max_file_size:
                big_file_list.append(cur_path)
            else:
                multi_process_file_list.append(cur_path)

    if multi_process_file_list:
        ret = convert_obj._do_multi_process(multi_process_file_list)
        if ret != utils.VECTOR_COMPARISON_NONE_ERROR:
            return_code = ret
    for big_file in big_file_list:
        ret, _ = convert_obj.convert_format_for_one_file(big_file)
        convert_obj._handle_result_callback([ret, big_file])
        if ret != utils.VECTOR_COMPARISON_NONE_ERROR:
            return_code = ret

    if return_code != utils.VECTOR_COMPARISON_NONE_ERROR:
        error_file_path = os.path.join(
            convert_obj.output_path, utils.CONVERT_FAILED_FILE_LIST_NAME)
        if os.path.exists(error_file_path):
            utils.print_info_log(
                'The list of file that failed to convert has been written to "' + error_file_path + '".')
    # pylint: enable=W0212
    return return_code

if __name__ == "__main__":
    convert_parser = argparse.ArgumentParser()
    convert_parser.add_argument(
        '-d', '--dump_file', dest='dump_path', default='', required=True)
    convert_parser.add_argument(
        '-l', '--file_list', nargs="*", dest='file_list', default='')
    convert_parser.add_argument('-f', '--format', dest='format', default=None)
    convert_parser.add_argument(
        '-v', '--version', dest='dump_version', choices=[1, 2], type=int, default=2)
    convert_parser.add_argument('-s', '--shape', dest='shape', default=None)
    convert_parser.add_argument('-o', '--output_tensor',
                                dest='output', default=None)
    convert_parser.add_argument('-i', '--input_tensor', dest='input', default=None)
    convert_parser.add_argument(
        '-c', '--custom_script_path', dest='custom_script_path', default=None)
    convert_parser.add_argument('-out', '--output', dest='output_path', default='')
    convert_parser.add_argument(
        '-t', '--type', dest='output_file_type', choices=['npy', 'bin'], default='npy')

    args = convert_parser.parse_args()
    dump_failed = os.path.abspath(args.dump_path) + "/convert_failed_file_list.txt"
    if os.path.exists(dump_failed):
        os.remove(dump_failed)
    file_list = args.file_list
    if args.format is not None:
        convert = FormatConversionMain(args)
    else:
        convert = DumpDataParser(args)
    if args.file_list == "":
        file_list = os.listdir(args.dump_path)
    handle_multi_process(convert, file_list)
    if os.path.exists(dump_failed):
        with open(dump_failed, newline='') as failed_ops:
            file_reader = csv.reader(failed_ops, delimiter=',')
            file_list = [os.path.abspath(row[0]) for row in file_reader]
        args.format = None
        convert = DumpDataParser(args)
        handle_multi_process(convert, file_list)
