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
log test
"""
import logging
import os
import re
import shutil
import sys
import time


def test_log_stdout():
    # Clean up environment variables
    _rm_env_config()
    # print the logs without raising an exception.
    from mindspore import log as logger
    log_str = 'print informations'
    logger.error("1 test log message info :%s", log_str)
    logger.info("2 test log message info")
    logger.warning("3 test log message warning")
    logger.debug("4 test log message debug:%s", log_str)
    # Clean up _global_logger to avoid affecting for next usecase
    _clear_logger(logger)


def test_log_default():
    _rm_env_config()
    from mindspore import log as logger
    configdict = logger.get_log_config()
    targetdict = {'GLOG_v': '2', 'GLOG_logtostderr': '1'}
    assert configdict == targetdict
    # Clean up _global_logger to avoid affecting for next usecase
    _clear_logger(logger)


def test_log_setlevel():
    _rm_env_config()
    os.environ['GLOG_v'] = '0'
    from mindspore import log as logger
    # logger_instance = logger._get_logger()
    # del logger_instance
    loglevel = logger.get_level()
    log_str = 'print debug informations'
    logger.debug("5 test log message debug:%s", log_str)
    assert loglevel == '0'
    # Clean up _global_logger to avoid affecting for next usecase
    _clear_logger(logger)


def test_log_file():
    """
    test the log content written in log file
    """
    _rm_env_config()
    file_path = '/tmp/log/mindspore_test'
    os.environ['GLOG_v'] = '2'
    os.environ['GLOG_logtostderr'] = '0'
    os.environ['GLOG_log_dir'] = file_path
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
    filename = ''
    os.makedirs(file_path, exist_ok=True)
    from mindspore import log as logger
    logger.warning("test log message warning")
    f_list = os.listdir(file_path)
    # print f_list
    for file_name in f_list:
        if file_name.startswith('mindspore.log'):
            filename = f'{file_path}/{file_name}'
    cmd = f'cat {filename}'
    result = os.popen(cmd).read()
    # pylint: disable=anomalous-backslash-in-string

    pattern = "\[WARNING\] ME\(.*[0-9]:.*[0-9]\,.*[a-zA-Z0-9]\):.* " \
              "\[.*:.*[0-9]\] test log message warning"
    match_obj = re.match(pattern, result)
    # Clear test file
    if os.path.exists(file_path):
        shutil.rmtree(file_path)

    assert match_obj
    # Clean up _global_logger to avoid affecting for next usecase
    _clear_logger(logger)

def test_log_backup_count():
    """
    test backup count
    """
    # logger.reset_log_config(level=logging.INFO, console=False,
    #                        filepath=file_path, maxBytes=1000, backupCount=10)
    _rm_env_config()
    file_path = '/tmp/log/mindspore_test'
    os.environ['GLOG_v'] = '1'
    os.environ['GLOG_logtostderr'] = '0'
    os.environ['GLOG_log_dir'] = file_path
    os.environ['logger_maxBytes'] = '1000'
    os.environ['logger_backupCount'] = '10'

    if os.path.exists(file_path):
        shutil.rmtree(file_path)
    os.makedirs(file_path, exist_ok=True)
    from mindspore import log as logger

    log_count = 100
    for i in range(0, log_count, 1):
        logger.warning("test log message warning %r", i)

    cmd = f'cd {file_path};ls -l | grep \'mindspore.log.*\'|wc -l'
    backup_count = '11'
    file_count = os.popen(cmd).read().strip()

    if os.path.exists(file_path):
        shutil.rmtree(file_path)
    assert file_count == backup_count
    # Clean up _global_logger to avoid affecting for next usecase
    _clear_logger(logger)


def test_log_verify_envconfig():
    """
    test reset config
    """
    dictlist = []
    from mindspore import log as logger
    file_path = '/tmp'

    # level is not a number
    _rm_env_config()
    os.environ['GLOG_v'] = 'test'
    verify_dict_0 = logger._get_env_config()

    # level is not in range
    _rm_env_config()
    os.environ['GLOG_v'] = '100'
    verify_dict_1 = logger._get_env_config()

    # console is not a number
    _rm_env_config()
    os.environ['GLOG_logtostderr'] = 'test'
    verify_dict_2 = logger._get_env_config()

    # console is not in range
    _rm_env_config()
    os.environ['GLOG_logtostderr'] = '6'
    verify_dict_3 = logger._get_env_config()

    # path does not exist
    _rm_env_config()
    os.environ['GLOG_logtostderr'] = '0'
    os.environ['GLOG_log_dir'] = '/test'
    verify_dict_4 = logger._get_env_config()

    # path is not configured
    _rm_env_config()
    os.environ['GLOG_logtostderr'] = '0'
    verify_dict_5 = logger._get_env_config()

    # logger_maxBytes is not a number
    _rm_env_config()
    os.environ['GLOG_logtostderr'] = '0'
    os.environ['GLOG_log_dir'] = '/tmp'
    os.environ['logger_maxBytes'] = 'test'
    os.environ['logger_backupCount'] = '10'
    verify_dict_6 = logger._get_env_config()

    # logger_maxBytes is a negative number
    _rm_env_config()
    os.environ['GLOG_logtostderr'] = '0'
    os.environ['GLOG_log_dir'] = '/tmp'
    os.environ['logger_maxBytes'] = '-1'
    os.environ['logger_backupCount'] = '10'
    verify_dict_7 = logger._get_env_config()

    # logger_backupCount is not a number
    _rm_env_config()
    os.environ['GLOG_logtostderr'] = '0'
    os.environ['GLOG_log_dir'] = '/tmp'
    os.environ['logger_maxBytes'] = '0'
    os.environ['logger_backupCount'] = 'test'
    verify_dict_8 = logger._get_env_config()

    # logger_backupCount is a negative number
    _rm_env_config()
    os.environ['GLOG_logtostderr'] = '0'
    os.environ['GLOG_log_dir'] = '/tmp'
    os.environ['logger_maxBytes'] = '0'
    os.environ['logger_backupCount'] = '-1'
    verify_dict_9 = logger._get_env_config()

    for i in range(0, 10, 1):
        variable_name = f'verify_dict_{i}'
        dictlist.append(locals()[variable_name])

    for verify_dict in dictlist:
        try:
            logger._verify_config(verify_dict)
        except ValueError as ve:
            print(ve)
            assert True
        except TypeError as te:
            print(te)
            assert True
        else:
            assert False
    # Clean up _global_logger to avoid affecting for next usecase
    _clear_logger(logger)


def test_log_repeated_print():
    """
    test Log repeated printing
    # Print one log is right, otherwise error
    """
    _rm_env_config()
    from mindspore import log as logger
    py_logging = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    py_logging.addHandler(handler)
    logger.info("test log message info test ")
    # Clean up _global_logger to avoid affecting for next usecase
    _clear_logger(logger)


def test_log_getconfig():
    _rm_env_config()
    os.environ['GLOG_v'] = '3'
    os.environ['GLOG_logtostderr'] = '0'
    os.environ['GLOG_log_dir'] = '/tmp/log/'
    os.environ['logger_maxBytes'] = '1000'
    os.environ['logger_backupCount'] = '10'
    from mindspore import log as logger
    logger.info("test log message info test ")
    configdict = logger.get_log_config()
    targetdict = {'GLOG_v': '3', 'GLOG_log_dir': '/tmp/log',
                  'GLOG_logtostderr': '0', 'logger_maxBytes': 1000,
                  'logger_backupCount': 10, 'GLOG_stderrthreshold': '2'}
    assert configdict == targetdict
    # Clean up _global_logger to avoid affecting for next usecase
    _clear_logger(logger)


def test_log_perf():
    """
    Performance test with python logging
    """
    _rm_env_config()
    os.environ['GLOG_v'] = '3'
    from mindspore import log as logger
    loglevel = logging.ERROR
    logging.basicConfig()
    py_logging = logging.getLogger()
    py_logging.setLevel(loglevel)

    log_count = 100000

    print("logger level:", logger.get_level())
    print("py_logging level:", py_logging.getEffectiveLevel())

    # Calculate PY logging execution time
    start_time_py_logging = int(round(time.time() * 1000))

    for i in range(0, log_count, 1):
        py_logging.info("test log message info :%r", i)

    end_time_py_logging = int(round(time.time() * 1000))
    time_diff_py_logging = end_time_py_logging - start_time_py_logging

    # Calculate MS logger execution time
    start_time_logger = int(round(time.time() * 1000))

    for i in range(0, log_count, 1):
        logger.info("test log message info :%r", i)

    end_time_logger = int(round(time.time() * 1000))
    time_diff_logger = end_time_logger - start_time_logger

    # Calculate time difference
    time_diff = time_diff_logger - time_diff_py_logging
    strprint = f'time difference between MS logger ' \
               f'and Python logging: {time_diff} ms'
    print(strprint)
    std_time = 2000
    assert time_diff < std_time
    # Clean up _global_logger to avoid affecting for next usecase
    _clear_logger(logger)


def test_log_ms_import():
    _rm_env_config()
    import mindspore as ms
    configdict = ms.get_log_config()
    targetdict = {'GLOG_v': '2', 'GLOG_logtostderr': '1'}
    level = ms.get_level()
    assert configdict == targetdict and level == '2'


def _clear_logger(logger):
    if logger._global_logger:
        for handler in logger._global_logger.handlers:
            logger._global_logger.removeHandler(handler)
        logger._global_logger = None


def _rm_env_config():
    envlist = ['GLOG_v', 'GLOG_logtostderr', 'GLOG_log_dir', 'logger_maxBytes', 'logger_backupCount']
    for env in envlist:
        if os.environ.get(env):
            del os.environ[env]
