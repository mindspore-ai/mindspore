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
import sys
import os
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
_global_logger = None

# The flag for enable console output
_std_on = '1'
# The flag for disable console output
_std_off = '0'
# Rotating max bytes, default is 50M
_logger_def_max_bytes = '52428800'
# Rotating backup count, default is 30
_logger_def_backup_count = '30'
# The default log level
_logger_def_level = '2'

# Log level name and level mapping
_name_to_level = {
    'ERROR': 40,
    'WARNING': 30,
    'INFO': 20,
    'DEBUG': 10,
}

# GLog level and level name
_gloglevel_to_name = {
    '3': 'ERROR',
    '2': 'WARNING',
    '1': 'INFO',
    '0': 'DEBUG',
}

# The mapping of logger configurations to glog configurations
_confmap_dict = {'level': 'GLOG_v', 'console': 'GLOG_logtostderr', 'filepath': 'GLOG_log_dir',
                 'maxBytes': 'logger_maxBytes', 'backupCount': 'logger_backupCount',
                 'stderr_level': 'GLOG_stderrthreshold'}


class _MultiCompatibleRotatingFileHandler(RotatingFileHandler):
    """Inherit RotatingFileHandler for multiprocess compatibility."""

    def rolling_rename(self):
        """Rolling rename log files and set permission of Log file"""
        for i in range(self.backupCount - 1, 0, -1):
            sfn = self.rotation_filename("%s.%d" % (self.baseFilename, i))
            dfn = self.rotation_filename("%s.%d" % (self.baseFilename, i + 1))
            if os.path.exists(sfn):
                if os.path.exists(dfn):
                    os.remove(dfn)
                # Modify the permission of Log file
                os.chmod(sfn, stat.S_IREAD)
                os.rename(sfn, dfn)

    def doRollover(self):
        """Override doRollover for multiprocess compatibility
        and setting permission of Log file"""
        if self.stream:
            self.stream.close()
            self.stream = None

        # Attain an exclusive lock with blocking mode by `fcntl` module.
        with open(self.baseFilename, 'a') as file_pointer:
            if platform.system() != "Windows":
                fcntl.lockf(file_pointer.fileno(), fcntl.LOCK_EX)

        if self.backupCount > 0:
            self.rolling_rename()

        dfn = self.rotation_filename(self.baseFilename + ".1")
        if os.path.exists(dfn):
            os.remove(dfn)
        # Modify the permission of Log file
        os.chmod(self.baseFilename, stat.S_IREAD)
        self.rotate(self.baseFilename, dfn)

        with open(self.baseFilename, 'a'):
            # Modify the permission of Log file
            os.chmod(self.baseFilename, stat.S_IREAD | stat.S_IWRITE)

        if not self.delay:
            self.stream = self._open()


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
    if _global_logger:
        return _global_logger

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
    kwargs['level'] = _gloglevel_to_name.get(kwargs.get('level', _logger_def_level))
    kwargs['stderr_level'] = _gloglevel_to_name.get(kwargs.get('stderr_level', _logger_def_level))
    kwargs['console'] = not kwargs.get('console') == _std_off
    kwargs['maxBytes'] = int(kwargs.get('maxBytes', _logger_def_max_bytes))
    kwargs['backupCount'] = int(kwargs.get('backupCount', _logger_def_backup_count))
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


def get_level():
    """
    Get the logger level.

    Returns:
        str, the Log level includes 3(ERROR), 2(WARNING), 1(INFO), 0(DEBUG).

    Examples:
        >>> import os
        >>> os.environ['GLOG_v'] = '0'
        >>> from mindspore import log as logger
        >>> logger.get_level()
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
        if not console.isdigit() or console not in (_std_off, _std_on):
            raise ValueError(f'Incorrect value, The value of {_confmap_dict["console"]} must be 0 or 1,'
                             f' Output log to console, configure to 1.')

        if console == _std_off and not file_path:
            raise ValueError(f'When {_confmap_dict["console"]} is set to 0, The directory of '
                             f'saving log must be set, {_confmap_dict["filepath"]} cannot be empty.')

        # Check the input value of filepath
        if console == _std_off and file_path is not None:
            file_real_path = os.path.realpath(file_path)
            if not os.path.exists(file_real_path):
                raise ValueError(f'The file path does not exist. '
                                 f'{_confmap_dict["filepath"]}:{file_path}')

        # Check the input value of maxBytes
        max_bytes = kwargs.get('maxBytes', None)
        if console == _std_off and max_bytes is not None:
            if not max_bytes.isdigit():
                raise ValueError(f'Incorrect value, The value of {_confmap_dict["maxBytes"]} must be positive integer. '
                                 f'{_confmap_dict["maxBytes"]}:{max_bytes}')

        # Check the input value of backupCount
        backup_count = kwargs.get('backupCount', None)
        if console == _std_off and backup_count is not None:
            if not backup_count.isdigit():
                raise ValueError(f'Incorrect value, The value of {_confmap_dict["backupCount"]} must be positive '
                                 f'integer. {_confmap_dict["backupCount"]}:{backup_count}')


def _verify_level(level):
    """
    Verify log level.

    Args:
        level (str): The log level.
    """
    level_name = _gloglevel_to_name.get(level, None)

    # Check the value of input level
    if level_name not in _name_to_level:
        raise ValueError(f'Incorrect log level:{level}, Please check the configuration of GLOG_v or '
                         f'GLOG_stderrthreshold, desired log level :{_gloglevel_to_name}')


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
        >>> from mindspore import log as logger
        >>> logger.get_log_config()
    """
    logger = _get_logger()
    handler = logger.handlers[0]
    config_dict = {}
    config_dict['GLOG_v'] = get_level()
    config_dict['GLOG_logtostderr'] = _std_on

    if handler.name == 'FileHandler':
        config_dict['GLOG_logtostderr'] = _std_off
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
    sinfo = None
    stack_prefix = 'Stack (most recent call last):\n'
    sinfo = stack_prefix + "".join(traceback.format_stack(frame))
    return sinfo


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

    global _global_logger

    _setup_logger_lock.acquire()
    try:
        if _global_logger:
            return _global_logger

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
            logfile_dir = os.path.realpath(kwargs.get('filepath'))
            file_name = f'{logfile_dir}/{log_name}'
            logfile_handler = _MultiCompatibleRotatingFileHandler(
                filename=file_name,
                # Rotating max bytes, default is 50M
                maxBytes=kwargs.get('maxBytes', _logger_def_max_bytes),
                # Rotating backup count, default is 30
                backupCount=kwargs.get('backupCount', _logger_def_backup_count),
                encoding='utf8'
            )
            logfile_handler.name = 'FileHandler'
            logfile_handler.formatter = _DataFormatter(sub_module, formatter)
            logger.addHandler(logfile_handler)

            # Write the file and output warning and error logs to stderr
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.name = 'StreamHandler'
            console_handler.formatter = _DataFormatter(sub_module, formatter)
            console_handler.setLevel(kwargs.get('stderr_level', logging.WARNING))
            logger.addHandler(console_handler)

        _global_logger = logger

    finally:
        _setup_logger_lock.release()
    return _global_logger
