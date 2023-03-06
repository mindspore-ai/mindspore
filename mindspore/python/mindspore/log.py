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
"""
log module
"""
import warnings
import sys
import os
import re
import stat
import time
import logging
from logging.handlers import RotatingFileHandler
import traceback
import threading
import platform

if platform.system() != "Windows":
    import fcntl

__all__ = ['get_level', 'get_log_config']

# The lock for setting up the logger
_setup_logger_lock = threading.Lock()

# When getting the logger, Used to check whether
# the logger already exists
GLOBAL_LOGGER = None

# The flag for enable console output
STD_ON = '1'
# The flag for disable console output
STD_OFF = '0'
# Rotating max bytes, default is 50M
MAX_BYTES = '52428800'
# Rotating backup count, default is 30
BACKUP_COUNT = '30'
# The default log level
LOGGER_LEVEL = '2'

# Log level name and level mapping
_name_to_level = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50,
}

# GLog level and level name
_gloglevel_to_name = {
    '0': 'DEBUG',
    '1': 'INFO',
    '2': 'WARNING',
    '3': 'ERROR',
    '4': 'CRITICAL',
}

# The mapping of logger configurations to glog configurations
_confmap_dict = {'level': 'GLOG_v', 'console': 'GLOG_logtostderr', 'filepath': 'GLOG_log_dir',
                 'maxBytes': 'logger_maxBytes', 'backupCount': 'logger_backupCount',
                 'stderr_level': 'GLOG_stderrthreshold'}


class _MultiCompatibleRotatingFileHandler(RotatingFileHandler):
    """Inherit RotatingFileHandler for multiprocess compatibility."""

    def doRollover(self):
        """Override doRollover for multiprocess compatibility
        and setting permission of Log file"""

        # Attain an exclusive lock with blocking mode by `fcntl` module.
        with open(self.baseFilename, 'a') as file_pointer:
            if platform.system() != "Windows":
                fcntl.lockf(file_pointer.fileno(), fcntl.LOCK_EX)
            os.chmod(self.baseFilename, stat.S_IREAD)
            super().doRollover()
            # Modify the permission of Log file
            os.chmod(self.baseFilename, stat.S_IREAD | stat.S_IWRITE)


class _DataFormatter(logging.Formatter):
    """Log formatter"""

    def __init__(self, sub_module, fmt=None, **kwargs):
        """
        Initialization of logFormatter.

        Args:
            sub_module (str): The submodule name.
            fmt (str): Specified format pattern. Default: None.
        """
        super(_DataFormatter, self).__init__(fmt=fmt, **kwargs)
        self.sub_module = sub_module.upper()

    def formatTime(self, record, datefmt=None):
        """
        Override formatTime for uniform format %Y-%m-%d-%H:%M:%S.SSS.SSS

        Args:
            record (str): Log record.
            datefmt (str): Date format.

        Returns:
            str, formatted timestamp.
        """
        created_time = self.converter(record.created)
        if datefmt:
            return time.strftime(datefmt, created_time)

        timestamp = time.strftime('%Y-%m-%d-%H:%M:%S', created_time)
        msecs = str(round(record.msecs * 1000))
        # Format the time stamp
        return f'{timestamp}.{msecs[:3]}.{msecs[3:]}'

    def format(self, record):
        """
        Apply log format with specified pattern.

        Args:
            record (str): Format pattern.

        Returns:
            str, formatted log content according to format pattern.
        """
        # NOTICE: when the Installation directory of mindspore changed,
        # ms_home_path must be changed
        ms_install_home_path = 'mindspore'
        idx = record.pathname.rfind(ms_install_home_path)
        if idx >= 0:
            # Get the relative path of the file
            record.filepath = record.pathname[idx:]
        else:
            record.filepath = record.pathname
        record.sub_module = self.sub_module
        return super().format(record)


def _get_logger():
    """
    Get logger instance.

    Returns:
        Logger, a logger.
    """
    if GLOBAL_LOGGER:
        return GLOBAL_LOGGER

    kwargs = _get_env_config()
    _verify_config(kwargs)
    logger = _setup_logger(_adapt_cfg(kwargs))
    return logger


def _adapt_cfg(kwargs):
    """
    Glog configurations converted to logger configurations.

    Args:
        kwargs (dict): The dictionary of log configurations.

            - console (str): Whether to output log to stdout.
            - level (str): Log level.
            - filepath (str): The path for saving logs, if console is false, a file path must be assigned.
            - maxBytes (str): The Maximum value of a log file for rotating, only valid if console is false.
            - backupCount (str): The count of rotating backup log files, only valid if console is false.

    Returns:
        Dict, the input parameter dictionary.
    """
    kwargs['level'] = _gloglevel_to_name.get(kwargs.get('level', LOGGER_LEVEL))
    kwargs['stderr_level'] = _gloglevel_to_name.get(kwargs.get('stderr_level', LOGGER_LEVEL))
    kwargs['console'] = not kwargs.get('console') == STD_OFF
    kwargs['maxBytes'] = int(kwargs.get('maxBytes', MAX_BYTES))
    kwargs['backupCount'] = int(kwargs.get('backupCount', BACKUP_COUNT))
    return kwargs


def info(msg, *args, **kwargs):
    """
    Log a message with severity 'INFO' on the MindSpore logger.

    Examples:
        >>> from mindspore import log as logger
        >>> logger.info("The arg(%s) is: %r", name, arg)
    """
    _get_logger().info(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    """
    Log a message with severity 'DEBUG' on the MindSpore logger.

    Examples:
        >>> from mindspore import log as logger
        >>> logger.debug("The arg(%s) is: %r", name, arg)
    """
    _get_logger().debug(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    """Log a message with severity 'ERROR' on the MindSpore logger."""
    _get_logger().error(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    """Log a message with severity 'WARNING' on the MindSpore logger."""
    _get_logger().warning(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    """Log a message with severity 'CRITICAL' on the MindSpore logger."""
    _get_logger().critical(msg, *args, **kwargs)


def get_level():
    """
    Get the logger level.

    Returns:
        str, the Log level includes 4(CRITICAL), 3(ERROR), 2(WARNING), 1(INFO), 0(DEBUG).

    Examples:
        >>> import os
        >>> os.environ['GLOG_v'] = '0'
        >>> from mindspore import log as logger
        >>> level = logger.get_level()
        >>> print(level)
        '0'
    """
    # level and glog level mapping dictionary
    level_to_glog_level = dict(zip(_name_to_level.values(), _gloglevel_to_name.keys()))

    return level_to_glog_level.get(_get_logger().getEffectiveLevel())


def _get_formatter():
    """
    Get the string of log formatter.

    Returns:
        str, the string of log formatter.
    """
    formatter = '[%(levelname)s] %(sub_module)s(%(process)d:' \
                '%(thread)d,%(processName)s):%(asctime)s ' \
                '[%(filepath)s:%(lineno)d] %(message)s'
    return formatter


def _get_env_config():
    """
    Get configurations from environment variables.

    Returns:
        Dict, the dictionary of configurations.
    """
    config_dict = {}
    for key, env_value in _confmap_dict.items():
        value = os.environ.get(env_value)
        if value:
            config_dict[key] = value.strip()
    return config_dict


def _check_directory_by_regular(target, reg=None, flag=re.ASCII, prim_name=None):
    """Check whether directory is legitimate."""
    if not isinstance(target, str):
        raise ValueError("The directory {} must be string, but got {}, please check it".format(target, type(target)))
    if reg is None:
        reg = r"^[\/0-9a-zA-Z@\_\-\.\:\\]+$"
    if re.match(reg, target, flag) is None:
        prim_name = f'in `{prim_name}`' if prim_name else ""
        raise ValueError("'{}' {} is illegal, it should be match regular'{}' by flag'{}'".format(
            target, prim_name, reg, flag))


def _make_directory(path: str):
    """Make directory."""
    if path is None or not isinstance(path, str) or path.strip() == "":
        raise TypeError("Input path '{}' is invalid type".format(path))

    path = os.path.realpath(path)
    _check_directory_by_regular(path)
    if os.path.exists(path):
        real_path = path
    else:
        try:
            permissions = os.R_OK | os.W_OK | os.X_OK
            os.umask(permissions << 3 | permissions)
            mode = permissions << 6
            os.makedirs(path, mode=mode, exist_ok=True)
            real_path = path
        except PermissionError:
            raise TypeError("No write permission on the directory `{path}`.")
    return real_path


def _verify_config(kwargs):
    """
    Verify log configurations.

    Args:
        kwargs (dict): The dictionary of log configurations.

            - console (str): Whether to output log to stdout.
            - level (str): Log level.
            - filepath (str): The path for saving logs, if console is false, a file path must be assigned.
            - maxBytes (str): The Maximum value of a log file for rotating, only valid if console is false.
            - backupCount (str): The count of rotating backup log files, only valid if console is false.
    """
    # Check the input value of level
    level = kwargs.get('level', None)
    if level is not None:
        _verify_level(level)

    # Check the input value of stderr_level
    level = kwargs.get('stderr_level', None)
    if level is not None:
        _verify_level(level)

    # Check the input value of console
    console = kwargs.get('console', None)
    file_path = kwargs.get('filepath', None)

    if console is not None:
        if not console.isdigit() or console not in (STD_OFF, STD_ON):
            raise ValueError(f'Incorrect value, the value of {_confmap_dict["console"]} must be 0 or 1, '
                             f'but got {console}.')

        if console == STD_OFF and not file_path:
            raise ValueError(f'When {_confmap_dict["console"]} is set to 0, the directory of saving log '
                             f'{_confmap_dict["filepath"]} must be set, but got it empty.')

        # Check the input value of filepath
        if console == STD_OFF and file_path is not None:
            file_real_path = os.path.realpath(file_path)
            if not os.path.exists(file_real_path):
                _make_directory(file_real_path)
        # Check the input value of maxBytes
        max_bytes = kwargs.get('maxBytes', None)
        if console == STD_OFF and max_bytes is not None:
            if not max_bytes.isdigit():
                raise ValueError(f'Incorrect value, the value of {_confmap_dict["maxBytes"]} must be positive integer. '
                                 f'But got {_confmap_dict["maxBytes"]}:{max_bytes}.')

        # Check the input value of backupCount
        backup_count = kwargs.get('backupCount', None)
        if console == STD_OFF and backup_count is not None:
            if not backup_count.isdigit():
                raise ValueError(f'Incorrect value, the value of {_confmap_dict["backupCount"]} must be positive '
                                 f'integer. But got {_confmap_dict["backupCount"]}:{backup_count}')


def _verify_level(level):
    """
    Verify log level.

    Args:
        level (str): The log level.
    """
    level_name = _gloglevel_to_name.get(level, None)

    # Check the value of input level
    if level_name not in _name_to_level:
        raise ValueError(f'Incorrect log level, please check the configuration of GLOG_v or '
                         f'GLOG_stderrthreshold, desired log level: 4-CRITICAL, 3-ERROR, 2-WARNING, '
                         f'1-INFO, 0-DEBUG. But got {level}.')


def get_log_config():
    """
    Get logger configurations.

    Returns:
        Dict, the dictionary of logger configurations.

    Examples:
        >>> import os
        >>> os.environ['GLOG_v'] = '1'
        >>> os.environ['GLOG_logtostderr'] = '0'
        >>> os.environ['GLOG_log_dir'] = '/var/log'
        >>> os.environ['logger_maxBytes'] = '5242880'
        >>> os.environ['logger_backupCount'] = '10'
        >>> os.environ['GLOG_stderrthreshold'] = '2'
        >>> from mindspore import log as logger
        >>> config= logger.get_log_config()
        >>> print(config)
        {'GLOG_v': '1', 'GLOG_logtostderr': '0', 'GLOG_log_dir': '/var/log',
         'logger_maxBytes': '5242880', 'logger_backupCount': '10', 'GLOG_stderrthreshold': '2'}
    """
    logger = _get_logger()
    handler = logger.handlers[0]
    config_dict = {}
    config_dict['GLOG_v'] = get_level()
    config_dict['GLOG_logtostderr'] = STD_ON

    if handler.name == 'FileHandler':
        config_dict['GLOG_logtostderr'] = STD_OFF
        # Separating file path and name
        file_path_and_name = os.path.split(handler.baseFilename)
        config_dict['GLOG_log_dir'] = file_path_and_name[0]
        config_dict['logger_maxBytes'] = handler.maxBytes
        config_dict['logger_backupCount'] = handler.backupCount
        handler_stderr = logger.handlers[1]
        # level and glog level mapping dictionary
        level_to_glog_level = dict(zip(_name_to_level.values(), _gloglevel_to_name.keys()))
        config_dict['GLOG_stderrthreshold'] = level_to_glog_level.get(handler_stderr.level)
    return config_dict


def _clear_handler(logger):
    """Clear the handlers that has been set, avoid repeated loading"""
    for handler in logger.handlers:
        logger.removeHandler(handler)


def _find_caller(stack_info=False, stacklevel=1):
    """
    Find the stack frame of the caller.

    Override findCaller on the logger, Support for getting log record.
    Find the stack frame of the caller so that we can note the source
    file name, function name and line number.

    Args:
        stack_info (bool): If the value is true, print stack information to the log. Default: False.

    Returns:
        tuple, the tuple of the frame data.
    """
    f = sys._getframe(3)
    sinfo = None
    # log_file is used to check caller stack frame
    log_file = os.path.normcase(f.f_code.co_filename)
    f = f.f_back
    rv = "(unknown file)", 0, "(unknown function)", None
    while f:
        co = f.f_code
        filename = os.path.normcase(co.co_filename)
        if log_file == filename:
            f = f.f_back
            continue
        if stack_info:
            sinfo = _get_stack_info(f)
        rv = (co.co_filename, f.f_lineno, co.co_name, sinfo)
        break
    return rv


def _get_stack_info(frame):
    """
    Get the stack information.

    Args:
        frame(frame): the frame requiring information.

    Returns:
        str, the string of the stack information.
    """
    stack_prefix = 'Stack (most recent call last):\n'
    sinfo = stack_prefix + "".join(traceback.format_stack(frame))
    return sinfo


def _get_rank_id():
    """Get rank id."""
    rank_id = os.getenv('RANK_ID')
    gpu_rank_id = os.getenv('OMPI_COMM_WORLD_RANK')
    rank = '0'
    if rank_id and gpu_rank_id and rank_id != gpu_rank_id:
        warnings.warn(
            f"Environment variables RANK_ID and OMPI_COMM_WORLD_RANK set by different values, RANK_ID={rank_id}, "
            f"OMPI_COMM_WORLD_RANK={gpu_rank_id}. We will use RANK_ID to get rank id by default.")
    if rank_id:
        rank = rank_id
    elif gpu_rank_id:
        rank = gpu_rank_id
    return rank


def _create_logfile_dir(kwargs):
    """
    create logs dir
    Args:
        kwargs (dict): The dictionary of log configurations.
    Returns:
        Log_dir: Create subdirectory.
        Examples:
        >>> /rank_0/logs
    """
    log_dir = os.path.realpath(kwargs.get('filepath'))
    rank_id = _get_rank_id()
    log_dir += '/rank_' + rank_id + '/logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _setup_logger(kwargs):
    """
    Set up the logger.

    Args:
        kwargs (dict): The dictionary of log configurations.

            - console (bool): Whether to output log to stdout. Default: True.
            - level (str): Log level. Default: WARNING.
            - filepath (str): The path for saving logs, if console is false, a file path must be assigned.
            - maxBytes (int): The Maximum value of a log file for rotating, only valid if console is false.
              Default: 52428800.
            - backupCount (int): The count of rotating backup log files, only valid if console is false. Default: 30.

    Returns:
        Logger, well-configured logger.
    """

    # The name of Submodule
    sub_module = 'ME'
    # The name of Base log file
    pid = str(os.getpid())
    log_name = 'mindspore.log.' + pid

    global GLOBAL_LOGGER

    _setup_logger_lock.acquire()
    try:
        if GLOBAL_LOGGER:
            return GLOBAL_LOGGER

        logger = logging.getLogger(name=f'{sub_module}.{log_name}')
        # Override findCaller on the logger, Support for getting log record
        logger.findCaller = _find_caller
        console = kwargs.get('console', True)
        # Set log level
        logger.setLevel(kwargs.get('level', logging.WARNING))
        # Set "propagate" attribute to False, stop searching up the hierarchy,
        # avoid to load the handler of the root logger
        logger.propagate = False
        # Get the formatter for handler
        formatter = _get_formatter()

        # Clean up handle to avoid repeated loading
        _clear_handler(logger)

        # Set streamhandler for the console appender
        if console:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.name = 'StreamHandler'
            console_handler.formatter = _DataFormatter(sub_module, formatter)
            logger.addHandler(console_handler)

        # Set rotatingFileHandler for the file appender
        else:
            # filepath cannot be null, checked in function _verify_config ()
            logfile_dir = _create_logfile_dir(kwargs)
            file_name = f'{logfile_dir}/{log_name}'
            logfile_handler = _MultiCompatibleRotatingFileHandler(
                filename=file_name,
                # Rotating max bytes, default is 50M
                maxBytes=kwargs.get('maxBytes', MAX_BYTES),
                # Rotating backup count, default is 30
                backupCount=kwargs.get('backupCount', BACKUP_COUNT),
                encoding='utf8'
            )
            logfile_handler.name = 'FileHandler'
            logfile_handler.formatter = _DataFormatter(sub_module, formatter)
            logger.addHandler(logfile_handler)

            with open(file_name, 'a'):
                # Modify the permission of Log file
                os.chmod(file_name, stat.S_IREAD | stat.S_IWRITE)

            # Write the file and output warning and error logs to stderr
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.name = 'StreamHandler'
            console_handler.formatter = _DataFormatter(sub_module, formatter)
            console_handler.setLevel(kwargs.get('stderr_level', logging.WARNING))
            logger.addHandler(console_handler)

        GLOBAL_LOGGER = logger

    finally:
        _setup_logger_lock.release()
    return GLOBAL_LOGGER


class _LogActionOnce:
    """
    A wrapper for modify the warning logging to an empty function. This is used when we want to only log
    once to avoid the repeated logging.

    Args:
        logger (logging): The logger object.

    """
    is_logged = dict()

    def __init__(self, logger, key, no_warning=False):
        self.logger = logger
        self.key = key
        self.no_warning = no_warning

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            if not hasattr(self.logger, 'warning'):
                return func(*args, **kwargs)

            _old_ = self.logger.warning
            if self.no_warning or self.key in _LogActionOnce.is_logged:
                self.logger.warning = lambda x: x
            else:
                _LogActionOnce.is_logged[self.key] = True
            res = func(*args, **kwargs)
            if hasattr(self.logger, 'warning'):
                self.logger.warning = _old_
            return res

        return wrapper
