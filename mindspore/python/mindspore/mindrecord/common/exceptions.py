# Copyright 2019 Huawei Technologies Co., Ltd
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
Define custom exception

Error rule:
    EXCEPTIONS key: The error code key should be as same as the realized class name.
     exception No: common module error No is 1-99,
                        unknown error No is 0,
                    shard* api error No is 100-199
                    file* api error No is 200-299
     error message: It is the base error message.
                    You can recover error message in realized class __init__ function.
"""
from .enums import LogRuntime, ErrorCodeType, ErrorLevel
from .constant import SYS_ID
EXCEPTIONS = dict(
    # the format of list is [exception No, base error message]
    UnknownError=[0, 'Unknown Error.'],
    ParamTypeError=[1, 'Param type is error.'],
    ParamValueError=[2, 'Param value is error.'],
    ParamMissError=[3, 'Param missing.'],
    PathNotExistsError=[4, 'Path does not exist.'],
    DbConnectionError=[5, 'Db connection is error.'],

    # MindRecord error 100-199 for shard*
    MRMOpenError=[100, 'MindRecord could not open.'],
    MRMOpenForAppendError=[101, 'MindRecord could not open for append.'],
    MRMInvalidPageSizeError=[102, 'Failed to set page size.'],
    MRMInvalidHeaderSizeError=[103, 'Failed to set header size.'],
    MRMSetHeaderError=[104, 'Failed to set header.'],
    MRMWriteDatasetError=[105, 'Failed to write dataset.'],
    MRMCommitError=[107, 'Failed to commit.'],

    MRMLaunchError=[108, 'Failed to launch.'],
    MRMFinishError=[109, 'Failed to finish.'],
    MRMCloseError=[110, 'Failed to close.'],

    MRMAddSchemaError=[111, 'Failed to add schema.'],
    MRMAddIndexError=[112, 'Failed to add index field.'],
    MRMBuildSchemaError=[113, 'Failed to build schema.'],
    MRMGetMetaError=[114, 'Failed to get meta info.'],

    MRMIndexGeneratorError=[115, 'Failed to create index generator.'],
    MRMGenerateIndexError=[116, 'Failed to generate index.'],

    MRMInitSegmentError=[117, 'Failed to initialize segment.'],
    MRMFetchCandidateFieldsError=[118, 'Failed to fetch candidate category fields.'],
    MRMReadCategoryInfoError=[119, 'Failed to read category information.'],
    MRMFetchDataError=[120, 'Failed to fetch data by category.'],


    # MindRecord error 200-299 for File* and MindPage
    MRMInvalidSchemaError=[200, 'Schema is error.'],
    MRMValidateDataError=[201, 'Raw data is valid.'],
    MRMDefineIndexError=[202, 'Index field is error.'],
    MRMDefineBlobError=[203, 'Blob field is error.'],
    MRMUnsupportedSchemaError=[204, 'Schema is not supported.'],
    MRMDefineCategoryError=[205, 'Category field is error.'],

)


class MindRecordException(Exception):
    """MindRecord base error class."""

    def __init__(self):
        """Initialize an error which may occurs in mindrecord."""
        super(MindRecordException, self).__init__()
        class_name = self.__class__.__name__
        error_item = EXCEPTIONS.get(class_name) if class_name in EXCEPTIONS else EXCEPTIONS.get('UnknownError')
        self._error_msg = error_item[1]
        self._error_code = MindRecordException.transform_error_code(error_item[0])

    def __str__(self):
        return "[{}]: {}".format(self.__class__.__name__, self._error_msg)

    @property
    def error_msg(self):
        """return the description of this error."""
        return self._error_msg

    @error_msg.setter
    def error_msg(self, msg):
        self._error_msg = msg

    @property
    def error_code(self):
        """return the unique error number of this error."""
        return self._error_code

    @staticmethod
    def transform_error_code(exception_no):
        """
        Transform mindrecord exception no to GE error code.

        error_code = ((0xFF & runtime) << 30) \
                    | ((0xFF & error_code_type) << 28) \
                    | ((0xFF & error_level) << 25) \
                    | ((0xFF & sys_id) << 17) \
                    | ((0xFF & mod_id) << 12) \
                    | (0x0FFF & exception_no)
        Args:
            exception_no: Integer. Exception number.

        Returns:
            Integer, error code.

        """
        runtime = LogRuntime.RT_HOST
        error_code_type = ErrorCodeType.ERROR_CODE
        error_level = ErrorLevel.COMMON_LEVEL
        mod_id = int(exception_no / 100) + 1
        error_code = (((0xFF & runtime) << 30)
                      | ((0xFF & error_code_type) << 28)
                      | ((0xFF & error_level) << 25)
                      | ((0xFF & SYS_ID) << 17)
                      | ((0xFF & mod_id) << 12)
                      | (0x0FFF & exception_no))
        return error_code


class UnknownError(MindRecordException):
    """Raise an unknown error when an unknown error occurs."""


class ParamValueError(MindRecordException):
    """
    Request param value error.
    """

    def __init__(self, error_detail):
        super(ParamValueError, self).__init__()
        self.error_msg = 'Invalid parameter value. {}'.format(error_detail)


class ParamTypeError(MindRecordException):
    """
    Request param type error.
    """

    def __init__(self, param_name, expected_type):
        super(ParamTypeError, self).__init__()
        self.error_msg = "Invalid parameter type. '{}' expect {} type." \
                         "".format(param_name, expected_type)


class ParamMissError(MindRecordException):
    """
    missing param error.
    """

    def __init__(self, param_name):
        super(ParamMissError, self).__init__()
        self.error_msg = "Param missing. '{}' is required.".format(param_name)


class PathNotExistsError(MindRecordException):
    """
    invalid path.
    """
    def __init__(self, error_path):
        super(PathNotExistsError, self).__init__()
        self.error_msg = 'Invalid path. {}'.format(error_path)


class DbConnectionError(MindRecordException):
    """
    Database connection error.
    """
    def __init__(self, error_detail):
        super(DbConnectionError, self).__init__()
        self.error_msg = 'Db connection is error. Detail: {}'.format(error_detail)


class MRMOpenError(MindRecordException):
    """
    Raised when could not open mind record file successfully.
    """
    def __init__(self):
        super(MRMOpenError, self).__init__()
        self.error_msg = 'MindRecord File could not open successfully.'


class MRMOpenForAppendError(MindRecordException):
    """
    Raised when could not open mind record file successfully for append.
    """
    def __init__(self):
        super(MRMOpenForAppendError, self).__init__()
        self.error_msg = 'MindRecord File could not open successfully for append.'


class MRMInvalidPageSizeError(MindRecordException):
    pass


class MRMInvalidHeaderSizeError(MindRecordException):
    pass


class MRMSetHeaderError(MindRecordException):
    pass


class MRMWriteDatasetError(MindRecordException):
    pass


class MRMCommitError(MindRecordException):
    pass


class MRMLaunchError(MindRecordException):
    pass


class MRMFinishError(MindRecordException):
    pass


class MRMCloseError(MindRecordException):
    pass


class MRMAddSchemaError(MindRecordException):
    pass


class MRMAddIndexError(MindRecordException):
    pass


class MRMBuildSchemaError(MindRecordException):
    pass


class MRMGetMetaError(MindRecordException):
    pass


class MRMIndexGeneratorError(MindRecordException):
    pass


class MRMGenerateIndexError(MindRecordException):
    pass


class MRMInitSegmentError(MindRecordException):
    pass


class MRMFetchCandidateFieldsError(MindRecordException):
    pass


class MRMReadCategoryInfoError(MindRecordException):
    pass


class MRMFetchDataError(MindRecordException):
    pass


class MRMInvalidSchemaError(MindRecordException):
    def __init__(self, error_detail):
        super(MRMInvalidSchemaError, self).__init__()
        self.error_msg = 'Schema format is error. Detail: {}'.format(error_detail)


class MRMValidateDataError(MindRecordException):
    def __init__(self, error_detail):
        super(MRMValidateDataError, self).__init__()
        self.error_msg = 'Raw data do not match the schema. Detail: {}'.format(error_detail)


class MRMDefineIndexError(MindRecordException):
    def __init__(self, error_detail):
        super(MRMDefineIndexError, self).__init__()
        self.error_msg = 'Failed to define index field. Detail: {}'.format(error_detail)


class MRMDefineBlobError(MindRecordException):
    def __init__(self, error_detail):
        super(MRMDefineBlobError, self).__init__()
        self.error_msg = 'Failed to define blob field. Detail: {}'.format(error_detail)


class MRMUnsupportedSchemaError(MindRecordException):
    def __init__(self, error_detail):
        super(MRMUnsupportedSchemaError, self).__init__()
        self.error_msg = 'Schema is not supported. Detail: {}'.format(error_detail)


class MRMDefineCategoryError(MindRecordException):
    def __init__(self, error_detail):
        super(MRMDefineCategoryError, self).__init__()
        self.error_msg = 'Failed to define category field. Detail: {}'.format(error_detail)
