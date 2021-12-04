mindspore.get_level
======================

.. py:class:: mindspore.get_level()

    获取日志记录器的级别。

    **返回：**

    string，日志级别包括4(EXCEPTION)、3(ERROR)、2(WARNING)、1(INFO)和0(DEBUG)。

    **样例：**
    
    >>> import os
    >>> os.environ['GLOG_v'] = '0'
    >>> from mindspore import log as logger
    >>> level = logger.get_level()
    >>> print(level)
    '0'
    