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
"""
Module to provide conversion capabalities from .timestamp async dump files to .npy.
It's an internal module for debugger backend but not exposed to users.
"""
import os
import glob
import stat
import sys
from pathlib import Path
from importlib import import_module
from collections import namedtuple

import numpy as np


class ConvertToolLoader:
    """Module to load CANN conversion tool."""

    def __init__(self):
        self.utils = None
        self.common = None
        self.dump_data_parser = None
        self.format_conversion = None
        self.progress = None
        self.log = None
        self.compare_none_error = None
        self.compare_exception = None
        self.load_convert_tool()

    @staticmethod
    def find_toolkit_path():
        """Find the path to Ascend toolkit."""
        ascend_toolkit_path = os.getenv("ASCEND_TOOLKIT_PATH")
        if not ascend_toolkit_path:
            ascend_toolkit_path = "/usr/local/Ascend"
        if not os.path.exists(ascend_toolkit_path):
            raise ValueError(
                "Path {} does not exist. Please install Ascend run packages " \
                "and set the environment variable $ASCEND_TOOLKIT_PATH correctly.".format(ascend_toolkit_path))
        toolkit_search_path = Path(ascend_toolkit_path).resolve()
        msaccucmp_file_list = list(toolkit_search_path.rglob('msaccucmp.py*'))
        if not msaccucmp_file_list:
            toolkit_search_path = toolkit_search_path / 'tools'
            msaccucmp_file_list = list(toolkit_search_path.rglob('msaccucmp.py*'))
        if not msaccucmp_file_list:
            raise ValueError("Failed to find msaccucmp.py or msaccucmp.pyc file under {}. " \
                             "Please install Ascend toolkit.".format(ascend_toolkit_path))
        return msaccucmp_file_list[0].parent

    def load_convert_tool(self):
        """load CANN conversion tool from the toolkit path."""
        toolkit_path = self.find_toolkit_path()
        # add toolkit path to system searching module path
        if str(toolkit_path) not in sys.path:
            sys.path.append(str(toolkit_path))
        try:
            self.utils = import_module('utils')
            self.common = import_module('common')
            self.dump_data_parser = import_module(
                'dump_data_parser').DumpDataParser
            self.format_conversion = import_module(
                'shape_conversion').FormatConversionMain
        except ModuleNotFoundError:
            # restore system searching module path
            if str(toolkit_path) in sys.path:
                sys.path.remove(str(toolkit_path))
            raise ModuleNotFoundError(
                "Failed to load CANN conversion tools under {}. Please make sure Ascend " \
                "toolkit has been installed properly.".format(toolkit_path))

        try:
            self.progress = import_module("progress").Progress
        except (ModuleNotFoundError, AttributeError):
            self.progress = self.utils.Progress
        try:
            self.log = import_module("log")
            if not hasattr(self.log, "print_error_log"):
                raise ModuleNotFoundError
        except ModuleNotFoundError:
            self.log = self.utils
        try:
            compare_error = import_module("compare_error")
            self.compare_none_error = compare_error.CompareError.MSACCUCMP_NONE_ERROR
            self.compare_exception = compare_error.CompareError
        except ModuleNotFoundError:
            self.compare_none_error = self.utils.VECTOR_COMPARISON_NONE_ERROR
            self.compare_exception = self.utils.CompareError

        # restore system searching module path
        if str(toolkit_path) in sys.path:
            sys.path.remove(str(toolkit_path))


def parse_args(file_list, output_path):
    """Helper function to parse the input argument for the conversion configuration."""
    args_dict = dict()
    args_dict['dump_version'] = '2.0'
    args_dict['format'] = 'NCHW'
    args_dict['output_file_type'] = 'npy'
    args_dict['dump_path'] = output_path
    args_dict['output_path'] = output_path
    args_dict['file_list'] = file_list
    args_dict['input'] = None
    args_dict['output'] = None
    args_dict['shape'] = None
    args_dict['custom_script_path'] = None
    args_parser = namedtuple("args_parser", args_dict.keys())
    return args_parser(**args_dict)


class AsyncDumpConverter:
    """Convert the target async dump data into npy files."""

    def __init__(self, file_list, output_path):
        # check input path
        for file_item in file_list:
            file_item = os.path.realpath(file_item)
        output_path = os.path.realpath(output_path)

        self.convert_tool = ConvertToolLoader()
        self.args = parse_args(file_list, output_path)
        self.files_to_convert = self.args.file_list
        self.output_path = self.args.output_path
        self.failed_file_path = os.path.join(
            self.output_path, 'convert_failed_file_list.txt')
        self.clear_failed_list_file()

    def clear_failed_list_file(self):
        """Remove existing failed txt file."""
        if self.failed_file_path and os.path.exists(self.failed_file_path):
            os.remove(self.failed_file_path)

    def convert_files(self):
        """Main entry of the converter to convert async dump files into npy format."""
        self.convert_tool.log.print_info_log('Start to convert async dump files.')
        ret_code = self.convert_tool.compare_none_error
        if self.args.format is not None:
            convert = self.convert_tool.format_conversion(self.args)
        else:
            convert = self.convert_tool.dump_data_parser(self.args)
        ret_code = self.handle_multi_process(convert, self.files_to_convert)
        self._rename_generated_npy_files()
        if ret_code != self.convert_tool.compare_none_error:
            if os.path.exists(self.failed_file_path):
                self.convert_failed_tensors()
        self.convert_tool.log.print_info_log('Finish to convert async dump files.')

    def convert_failed_tensors(self):
        """Convert the failed tensor recorded in the failed txt file."""
        self.convert_tool.log.print_info_log(
            'Start to convert failed tensors recorded in ' + self.failed_file_path + '.')
        with open(self.failed_file_path) as failed_lines:
            for failed_line in failed_lines:
                try:
                    failed_line_list = failed_line.rstrip().split(',')
                    self.convert_one_failed_tensor(failed_line_list)
                except (ValueError, OSError, AttributeError, self.convert_tool.compare_exception) as err:
                    self.convert_tool.log.print_error_log(
                        'Failed to convert ' + failed_line + ' to Host format: ' + str(err))

    def convert_one_failed_tensor(self, failed_tensor):
        """Convert failed operator one by one."""
        if len(failed_tensor) <= 1:
            raise ValueError(
                "Invalid tensor info in convert_failed_file_list.txt")
        file_path = failed_tensor[0]
        type_index = failed_tensor[1:]
        op_data = self.convert_tool.utils.parse_dump_file(
            file_path, self.args.dump_version)
        for type_index_item in type_index:
            tensor_type, index = type_index_item.split(':')
            index = int(index)
            tensor = getattr(op_data, tensor_type)[index]
            dump_data_array = self.convert_tool.utils.deserialize_dump_data_to_array(tensor)
            array = dump_data_array.reshape(tensor.shape.dim)
            self._save_tensor_to_npy_file(
                file_path, tensor_type, index, tensor.format, array)

    def handle_multi_process(self, convert_obj, files):
        """Convert async format files to npy in a multithreaded manner."""
        return_code = self.convert_tool.compare_none_error
        # try looking for function in compatibility with the toolkit package version.
        progress = self.convert_tool.progress(len(files))
        if hasattr(convert_obj, 'multi_process'):
            _ = setattr(convert_obj.multi_process, '_progress', progress)
        else:
            _ = setattr(convert_obj, 'progress', progress)
        multi_process_file_list = []
        big_file_list = []
        max_file_size = 0
        if hasattr(convert_obj, 'multi_process'):
            max_file_size = getattr(convert_obj.multi_process, 'get_max_file_size')()
        else:
            max_file_size = getattr(convert_obj, '_get_max_file_size')()
        for cur_file in files:
            cur_path = cur_file
            if os.path.isfile(cur_path):
                if os.path.getsize(cur_path) > max_file_size:
                    big_file_list.append(cur_path)
                else:
                    multi_process_file_list.append(cur_path)
        if multi_process_file_list:
            ret_mp = self.convert_tool.compare_none_error
            if hasattr(convert_obj, 'multi_process'):
                ret_mp = getattr(convert_obj.multi_process, '_do_multi_process')(multi_process_file_list)
            else:
                ret_mp = getattr(convert_obj, '_do_multi_process')(multi_process_file_list)
            if ret_mp != self.convert_tool.compare_none_error:
                return_code = ret_mp
        for big_file in big_file_list:
            ret_bf = self.convert_tool.compare_none_error
            if hasattr(convert_obj, '_convert_format_for_one_file'):
                ret_bf, _ = getattr(convert_obj, '_convert_format_for_one_file')(big_file)
            else:
                ret_bf, _ = getattr(convert_obj, 'convert_format_for_one_file')(big_file)
            if hasattr(convert_obj, 'multi_process'):
                getattr(convert_obj.multi_process, '_handle_result_callback')([ret_bf, big_file])
            else:
                getattr(convert_obj, '_handle_result_callback')([ret_bf, big_file])
            if ret_bf != self.convert_tool.compare_none_error:
                return_code = ret_bf
        if return_code != self.convert_tool.compare_none_error:
            if os.path.exists(self.failed_file_path):
                self.convert_tool.log.print_info_log(
                    'The list of file that failed to convert has been written to "'
                    + self.failed_file_path + '".')
        return return_code

    def _save_tensor_to_npy_file(self, file_path, tensor_type, idx, tensor_format, dump_data_array):
        """Save tensor file into npy format."""
        file_name = os.path.basename(file_path)
        name_splits = file_name.split('.')
        name_splits[1] = name_splits[1].split('_')[-1]
        file_name_no_scope = '.'.join(name_splits)
        out_file_name = "%s.%s.%d.%s.npy" % (
            file_name_no_scope,
            tensor_type,
            idx,
            self.convert_tool.common.get_format_string(tensor_format)
        )
        out_path = os.path.join(self.output_path, out_file_name)
        np.save(out_path, dump_data_array)
        os.chmod(out_path, stat.S_IRUSR)

    def _rename_generated_npy_files(self):
        """In order to follow dump naming convention, rename npy files generated by CANN conversion tool."""
        target_file_list = []
        for in_file in self.files_to_convert:
            target_file_list.extend(glob.glob(in_file + "*.npy"))
        for target_file in target_file_list:
            old_filename = os.path.basename(target_file)
            name_splits = old_filename.split('.')
            name_splits[1] = name_splits[1].split('_')[-1]
            name_splits[-2] = self.args.format
            new_file_name = '.'.join(name_splits)
            out_path = os.path.join(self.output_path, new_file_name)
            os.rename(target_file, out_path)
            os.chmod(out_path, stat.S_IRUSR)
            self.convert_tool.log.print_info_log("Rename file " + target_file + " to " + out_path)
