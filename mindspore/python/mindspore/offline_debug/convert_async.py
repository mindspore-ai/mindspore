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
import sys
from pathlib import Path
from importlib import import_module
from collections import namedtuple
import multiprocessing


class ConvertToolLoader:
    """
    Module to load CANN conversion tool.
    """

    def __init__(self):
        self.utils = None
        self.common = None
        self.dump_data_parser = None
        self.format_conversion = None
        self.progress = None
        self.log = None
        self.compare_none_error = None
        self.compare_exception = None
        self.toolkit_path = self.find_toolkit_path()
        self.load_convert_tool()

    @staticmethod
    def find_toolkit_path():
        """
        Find the path to Ascend toolkit.
        """
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
        """
        Load CANN conversion tool from the toolkit path.
        """
        # add toolkit path to system searching module path
        if str(self.toolkit_path) not in sys.path:
            sys.path.insert(0, str(self.toolkit_path))
        try:
            self.utils = import_module('utils')
            self.common = import_module('common')
            self.dump_data_parser = import_module(
                'dump_data_parser').DumpDataParser
            self.format_conversion = import_module(
                'shape_conversion').FormatConversionMain
        except ModuleNotFoundError as err:
            self.reset_system_path()
            raise ModuleNotFoundError(
                "Failed to load CANN conversion tools under {}. Please make sure Ascend " \
                "toolkit has been installed properly.".format(self.toolkit_path)) from err

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

    def reset_system_path(self):
        """
        Restore system searching module path
        """
        if str(self.toolkit_path) in sys.path:
            sys.path.remove(str(self.toolkit_path))


def parse_args(file_list, output_path):
    """
    Helper function to parse the input argument for the conversion configuration.
    """
    args_dict = dict()
    args_dict['dump_version'] = '2.0'
    args_dict['format'] = 'NCHW'
    args_dict['output_file_type'] = 'msnpy'
    args_dict['dump_path'] = output_path
    args_dict['output_path'] = output_path
    args_dict['file_list'] = file_list
    args_dict['input'] = None
    args_dict['output'] = None
    args_dict['shape'] = None
    args_dict['custom_script_path'] = None
    args_parser = namedtuple("args_parser", list(args_dict.keys()))
    return args_parser(**args_dict)


class AsyncDumpConverter:
    """
    Convert the target async dump data into npy files.
    """

    def __init__(self, file_list, output_path):
        # check input path
        file_list = [os.path.realpath(file_item) for file_item in file_list]
        output_path = os.path.realpath(output_path)

        self.convert_tool = ConvertToolLoader()
        self.args = parse_args(file_list, output_path)
        self.files_to_convert = self.args.file_list
        self.output_path = self.args.output_path
        self.failed_file_path = os.path.join(
            self.output_path, 'convert_failed_file_list.txt')
        self.clear_failed_list_file()

    @staticmethod
    def _get_file_list(files, convert_obj):
        """
        Process to get file lists in multi_process.
        """
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
        return multi_process_file_list, big_file_list

    @staticmethod
    def _process_func(convert_obj):
        """
        get function to process format transformation.
        """
        if hasattr(convert_obj, '_convert_format_for_one_file'):
            func = getattr(convert_obj, '_convert_format_for_one_file')
        else:
            func = getattr(convert_obj, 'convert_format_for_one_file')
        return func

    @staticmethod
    def _result_callback_func(convert_obj):
        """
        get result callback function.
        """
        if hasattr(convert_obj, 'multi_process'):
            func = getattr(convert_obj.multi_process, '_handle_result_callback')
        else:
            func = getattr(convert_obj, '_handle_result_callback')
        return func

    def clear_failed_list_file(self):
        """
        Remove existing failed txt file.
        """
        if self.failed_file_path and os.path.exists(self.failed_file_path):
            os.remove(self.failed_file_path)

    def convert_files(self):
        """
        Main entry of the converter to convert async dump files into npy format.
        """
        self.convert_tool.log.print_info_log('Start to convert async dump files.')
        try:
            if self.args.format is not None:
                convert = self.convert_tool.format_conversion(self.args)
            else:
                convert = self.convert_tool.dump_data_parser(self.args)
            # 1. check if arguments are valid
            convert.check_arguments_valid()
            # 2. convert format for dump data
            ret_code = self.handle_multi_process(convert, self.files_to_convert)
            if ret_code != self.convert_tool.compare_none_error:
                self.convert_tool.log.print_info_log('An error has occurred while converting format.')
        finally:
            # clean up sys.path no matter conversion is successful or not to avoid pollution
            self.convert_tool.reset_system_path()
        self.convert_tool.log.print_info_log('Finish to convert async dump files.')

    def handle_multi_process(self, convert_obj, files):
        """
        Convert async format files to npy in a multithreaded manner.
        """
        return_code = self.convert_tool.compare_none_error
        # try looking for function in compatibility with the toolkit package version.
        progress = self.convert_tool.progress(len(files))
        if hasattr(convert_obj, 'multi_process'):
            setattr(convert_obj.multi_process, '_progress', progress)
        else:
            setattr(convert_obj, 'progress', progress)
        multi_process_file_list, big_file_list = self._get_file_list(files, convert_obj)
        if multi_process_file_list:
            ret_mp = self._process_in_multi_process(multi_process_file_list, convert_obj)
            if ret_mp != self.convert_tool.compare_none_error:
                return_code = ret_mp
        if big_file_list:
            ret_bf = self._process_in_single_process(big_file_list, convert_obj)
            if ret_bf != self.convert_tool.compare_none_error:
                return_code = ret_bf
        if return_code != self.convert_tool.compare_none_error:
            if os.path.exists(self.failed_file_path):
                self.convert_tool.log.print_info_log(
                    'The list of file that failed to convert has been written to "'
                    + self.failed_file_path + '".')
        return return_code

    def _process_in_single_process(self, big_file_list, convert_obj):
        """
        Process big file in single process.
        """
        return_code = self.convert_tool.compare_none_error
        for big_file in big_file_list:
            ret_bf, _ = self._process_func(convert_obj)(big_file)
            self._result_callback_func(convert_obj)([ret_bf, big_file])
            if ret_bf != self.convert_tool.compare_none_error:
                return_code = ret_bf
        return return_code

    def _process_in_multi_process(self, file_list, convert_obj):
        """
        Process files in multi process.
        """
        cpu_count = int((multiprocessing.cpu_count() + 1) / 2)
        ctx = multiprocessing.get_context('forkserver')
        pool = ctx.Pool(cpu_count)
        all_task = []
        for cur_path in file_list:
            task = pool.apply_async(self._process_func(convert_obj),
                                    args=(cur_path,),
                                    callback=self._result_callback_func(convert_obj))
            all_task.append(task)
        pool.close()
        pool.join()
        Result = namedtuple('Result', ['ret_code', 'msg'])
        for task in all_task:
            result = Result._make(task.get())
            cur_ret = result.ret_code
            if cur_ret != self.convert_tool.compare_none_error:
                return cur_ret
        return self.convert_tool.compare_none_error
