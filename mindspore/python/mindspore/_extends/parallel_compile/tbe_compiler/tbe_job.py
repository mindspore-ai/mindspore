# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""tbe compile job definition"""
from __future__ import absolute_import
from datetime import datetime, timezone
import json
from enum import Enum


class JobType(Enum):
    """ Job Type """
    INITIALIZE_JOB = 'Initialize'
    FINALIZE_JOB = 'Finalize'
    CHECK_JOB = 'CheckSupport'
    SELECT_JOB = 'SelectFormat'
    PRECOMPILE_JOB = 'PreCompile'
    COMPILE_JOB = 'Compile'
    FUSION_COMPILE_JOB = 'FusionOpCompile'
    TUNE_JOB = 'Tune'
    QUERY_JOB = 'Query'


class LogLevel(Enum):
    """ Log Level """
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    ERROR_MANAGER = 4


class JobStatus(Enum):
    """ Job Status """
    JOB_INITIAL = "INITIAL"
    JOB_FAILED = "FAILED"
    JOB_SUCCESS = "SUCCESS"
    JOB_RUNNING = "RUNNING"


class LogMessage:
    """ Log message """

    def __init__(self, index, level, info):
        self.index = index
        self.level = level
        self.info = info


def _get_message(msg, args):
    """
    Return the message for this LogRecord.

    Return the message for this LogRecord after merging any user-supplied
    arguments with the message.
    """
    msg = str(msg)
    if args:
        msg = msg % args
    return str(datetime.now(tz=timezone.utc)) + ": " + msg


class TbeJob:
    """ Tbe compilation job """

    def __init__(self, source_id, job_id, job_type, content, fusion_op_name, json_str, sys_info):
        self.source_id = source_id
        self.id = job_id
        self.type = JobType(job_type)
        self.status = JobStatus.JOB_INITIAL
        self.content = content
        self.fusion_op_name = fusion_op_name
        self.result = ""
        self.process_info = []
        self.json_string = json_str
        self._sys_logger = sys_info["logger"]
        self.sys_offline_tune = sys_info["offline_tune"]
        self.sys_tune_dump_path = sys_info["tune_dump_path"]
        self.sys_para_debug_path = sys_info["para_debug_path"]
        # license info
        self.rl_tune_switch = sys_info["rl_tune_switch"]
        self.rl_tune_list = sys_info["rl_tune_list"]
        self.op_tune_switch = sys_info["op_tune_switch"]
        self.op_tune_list = sys_info["op_tune_list"]
        self.pass_list = sys_info["pass_list"]

        # soc info
        self.soc_version = sys_info["socVersion"]
        self.core_num = sys_info["coreNum"]
        self.op_bank_path = sys_info["op_bank_path"]

    def debug(self, msg, *args, **kwargs):
        """
        log debug level info
        :param msg:
        :param args:
        :return:
        """
        processed_msg = _get_message(msg, args)
        message = LogMessage(len(self.process_info), LogLevel.DEBUG, processed_msg)
        self.process_info.append(message)
        self._sys_logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        """
        log info level info
        :param msg:
        :param args:
        :return:
        """
        processed_msg = _get_message(msg, args)
        message = LogMessage(len(self.process_info), LogLevel.INFO, processed_msg)
        self.process_info.append(message)
        self._sys_logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        """
        log warning level info
        :param msg:
        :param args:
        :return:
        """
        processed_msg = _get_message(msg, args)
        message = LogMessage(len(self.process_info), LogLevel.WARNING, processed_msg)
        self.process_info.append(message)
        self._sys_logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """
        log error level info
        :param msg:
        :param args:
        :return:
        """
        processed_msg = _get_message(msg, args)
        message = LogMessage(len(self.process_info), LogLevel.ERROR, processed_msg)
        self.process_info.append(message)
        self._sys_logger.error(msg, *args, **kwargs)

    def error_manager(self, msg, *args, **kwargs):
        """
        log exception level info
        :param msg:
        :param args:
        :return:
        """
        if not msg:
            self.warning("Get empty error manager message, op_name: {}".format(self.fusion_op_name))
            return
        exception_info = None
        op_name = self.fusion_op_name
        if isinstance(msg, Exception):
            for arg in msg.args:
                if isinstance(arg, dict) and "errCode" in arg:
                    exception_info = arg
                    break
            if not exception_info:
                self.error("Exception message:{}".format(msg))
                return
        else:
            exception_info = msg[0]
            if len(msg) >= 2:
                op_name = msg[1]
        if not isinstance(exception_info, dict) or not exception_info:
            self.warning("Get illegal error manager message, op_name: {}".format(self.fusion_op_name))
            return
        exception_info["op_name"] = op_name
        processed_msg = json.dumps(exception_info)
        message = LogMessage(len(self.process_info), LogLevel.ERROR_MANAGER, processed_msg)
        self.process_info.append(message)
        self._sys_logger.exception(msg, *args, **kwargs)

    def get_result(self):
        """
        Get tht job process result string
        :return: job process result string
        """
        result = dict()
        result["status"] = self.status.value
        result["source_id"] = self.source_id
        result["job_id"] = self.id
        result["job_type"] = self.type.value
        result["fusion_op_name"] = self.fusion_op_name
        result["result"] = self.result
        process_info = []
        for info in self.process_info:
            msg = {"index": info.index, "level": info.level.value, "message": info.info}
            process_info.append(msg)
        result["process_info"] = process_info
        return json.dumps(result)
