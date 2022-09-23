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
# ============================================================================
"""Definition of error code and relative messages in profiler module."""
from mindspore.profiler.common.exceptions.error_code import ProfilerErrors, \
    ProfilerErrorMsg


class ProfilerException(Exception):
    """
    Base class for Profilier exception.

    Examples:
        >>> raise ProfilerException(GeneralErrors.PATH_NOT_EXISTS_ERROR, 'path not exists')
    """

    RUNTIME = 1
    TYPE = 1
    LEVEL = 0
    SYSID = 42

    def __init__(self, error, message, http_code=500):
        """
        Initialization of ProfilerException.

        Args:
            error (Enum): Error value for specified case.
            message (str): Description for exception.
            http_code (int): Http code for exception. Default is 500.
        """
        if isinstance(message, str):
            message = ' '.join(message.split())
        super(ProfilerException, self).__init__(message)
        self.error = error
        self.message = message
        self.http_code = http_code

    def __str__(self):
        return '[{}] code: {}, msg: {}'.format(self.__class__.__name__, self.error_code, self.message)

    @property
    def error_code(self):
        """
        Transform exception no to Profiler error code.

        code compose(4bytes):
        runtime 2bits, type 2bits, level 3bits, sysid 8bits, modid 5bits, value 12bits.

        num = ((0xFF & runtime) << 30) \
                | ((0xFF & type) << 28) \
                | ((0xFF & level) << 25) \
                | ((0xFF & sysid) << 17) \
                | ((0xFF & modid) << 12) \
                | (0x0FFF & value)

        Returns:
            str, Hex string representing the composed Profiler error code.
        """
        num = (((0xFF & self.RUNTIME) << 30)
               | ((0xFF & self.TYPE) << 28)
               | ((0xFF & self.LEVEL) << 25)
               | ((0xFF & self.SYSID) << 17)
               | ((0xFF & 6) << 12)
               | (0x0FFF & self.error.value))

        return hex(num)[2:].zfill(8).upper()


class ProfilerParamValueErrorException(ProfilerException):
    """The parameter value error in profiler module."""

    def __init__(self, msg):
        super(ProfilerParamValueErrorException, self).__init__(
            error=ProfilerErrors.PARAM_VALUE_ERROR,
            message=ProfilerErrorMsg.PARAM_VALUE_ERROR.value.format(msg),
            http_code=400
        )


class ProfilerPathErrorException(ProfilerException):
    """The path error in profiler module."""

    def __init__(self, msg):
        super(ProfilerPathErrorException, self).__init__(
            error=ProfilerErrors.PATH_ERROR,
            message=ProfilerErrorMsg.PATH_ERROR.value.format(msg),
            http_code=400
        )


class ProfilerParamTypeErrorException(ProfilerException):
    """The parameter type error in profiler module."""

    def __init__(self, msg):
        super(ProfilerParamTypeErrorException, self).__init__(
            error=ProfilerErrors.PARAM_TYPE_ERROR,
            message=ProfilerErrorMsg.PARAM_TYPE_ERROR.value.format(msg),
            http_code=400
        )


class ProfilerDirNotFoundException(ProfilerException):
    """The dir not found exception in profiler module."""

    def __init__(self, msg):
        super(ProfilerDirNotFoundException, self).__init__(
            error=ProfilerErrors.DIR_NOT_FOUND_ERROR,
            message=ProfilerErrorMsg.DIR_NOT_FOUND_ERROR.value.format(msg),
            http_code=400
        )


class ProfilerFileNotFoundException(ProfilerException):
    """The file not found exception in profiler module."""

    def __init__(self, msg):
        super(ProfilerFileNotFoundException, self).__init__(
            error=ProfilerErrors.FILE_NOT_FOUND_ERROR,
            message=ProfilerErrorMsg.FILE_NOT_FOUND_ERROR.value.format(msg),
            http_code=400
        )


class ProfilerIOException(ProfilerException):
    """The IO exception in profiler module."""

    def __init__(self):
        super(ProfilerIOException, self).__init__(
            error=ProfilerErrors.IO_ERROR,
            message=ProfilerErrorMsg.IO_ERROR.value,
            http_code=400
        )


class ProfilerDeviceIdMismatchException(ProfilerException):
    """The device id mismatch exception in profiler module."""

    def __init__(self):
        super(ProfilerDeviceIdMismatchException, self).__init__(
            error=ProfilerErrors.DEVICE_ID_MISMATCH_ERROR,
            message=ProfilerErrorMsg.DEVICE_ID_MISMATCH_ERROR.value,
            http_code=400
        )


class ProfilerRawFileException(ProfilerException):
    """The raw file exception in profiler module."""

    def __init__(self, msg):
        super(ProfilerRawFileException, self).__init__(
            error=ProfilerErrors.RAW_FILE_ERROR,
            message=ProfilerErrorMsg.RAW_FILE_ERROR.value.format(msg),
            http_code=400
        )


class ProfilerColumnNotExistException(ProfilerException):
    """The column does not exist exception in profiler module."""

    def __init__(self, msg):
        super(ProfilerColumnNotExistException, self).__init__(
            error=ProfilerErrors.COLUMN_NOT_EXIST_ERROR,
            message=ProfilerErrorMsg.COLUMN_NOT_EXIST_ERROR.value.format(msg),
            http_code=400
        )


class ProfilerAnalyserNotExistException(ProfilerException):
    """The analyser in profiler module."""

    def __init__(self, msg):
        super(ProfilerAnalyserNotExistException, self).__init__(
            error=ProfilerErrors.ANALYSER_NOT_EXIST_ERROR,
            message=ProfilerErrorMsg.ANALYSER_NOT_EXIST_ERROR.value.format(msg),
            http_code=400
        )


class ProfilerDeviceIdException(ProfilerException):
    """The parameter device_id error in profiler module."""

    def __init__(self, msg):
        super(ProfilerDeviceIdException, self).__init__(
            error=ProfilerErrors.DEVICE_ID_ERROR,
            message=ProfilerErrorMsg.DEIVICE_ID_ERROR.value.format(msg),
            http_code=400
        )


class ProfilerOpTypeException(ProfilerException):
    """The parameter op_type error in profiler module."""

    def __init__(self, msg):
        super(ProfilerOpTypeException, self).__init__(
            error=ProfilerErrors.OP_TYPE_ERROR,
            message=ProfilerErrorMsg.OP_TYPE_ERROR.value.format(msg),
            http_code=400
        )


class ProfilerSortConditionException(ProfilerException):
    """The parameter sort_condition error in profiler module."""

    def __init__(self, msg):
        super(ProfilerSortConditionException, self).__init__(
            error=ProfilerErrors.SORT_CONDITION_ERROR,
            message=ProfilerErrorMsg.SORT_CONDITION_ERROR.value.format(msg),
            http_code=400
        )


class ProfilerFilterConditionException(ProfilerException):
    """The parameter filer_condition error in profiler module."""

    def __init__(self, msg):
        super(ProfilerFilterConditionException, self).__init__(
            error=ProfilerErrors.FILTER_CONDITION_ERROR,
            message=ProfilerErrorMsg.FILTER_CONDITION_ERROR.value.format(msg),
            http_code=400
        )


class ProfilerGroupConditionException(ProfilerException):
    """The parameter group_condition error in profiler module."""

    def __init__(self, msg):
        super(ProfilerGroupConditionException, self).__init__(
            error=ProfilerErrors.GROUP_CONDITION_ERROR,
            message=ProfilerErrorMsg.GROUP_CONDITION_ERROR.value.format(msg),
            http_code=400
        )


class ProfilerColumnNotSupportSortException(ProfilerException):
    """The column does not support to sort error in profiler module."""

    def __init__(self, msg):
        super(ProfilerColumnNotSupportSortException, self).__init__(
            error=ProfilerErrors.COLUMN_NOT_SUPPORT_SORT_ERROR,
            message=ProfilerErrorMsg.COLUMN_NOT_SUPPORT_SORT_ERROR.value.format(msg),
            http_code=400
        )


class StepNumNotSupportedException(ProfilerException):
    """The step number error in profiler module."""

    def __init__(self, msg):
        super(StepNumNotSupportedException, self).__init__(
            error=ProfilerErrors.STEP_NUM_NOT_SUPPORTED_ERROR,
            message=ProfilerErrorMsg.STEP_NUM_NOT_SUPPORTED_ERROR.value.format(msg),
            http_code=400
        )


class JobIdMismatchException(ProfilerException):
    """The Job ID mismatch error in profiler module."""

    def __init__(self):
        super(JobIdMismatchException, self).__init__(
            error=ProfilerErrors.JOB_ID_MISMATCH_ERROR,
            message=ProfilerErrorMsg.JOB_ID_MISMATCH_ERROR.value,
            http_code=400
        )


class ProfilerPipelineOpNotExistException(ProfilerException):
    """The minddata pipeline operator does not exist error in profiler module."""

    def __init__(self, msg):
        super(ProfilerPipelineOpNotExistException, self).__init__(
            error=ProfilerErrors.PIPELINE_OP_NOT_EXIST_ERROR,
            message=ProfilerErrorMsg.PIPELINE_OP_NOT_EXIST_ERROR.value.format(msg),
            http_code=400
        )
