mindspore.get_log_config
=========================

.. py:class:: mindspore.get_log_config()

    获取日志配置。

    **返回：**

    Dict，日志配置字典。

    **样例：**

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
    