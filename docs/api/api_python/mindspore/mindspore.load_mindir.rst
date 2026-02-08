mindspore.load_mindir
=======================================

.. py:function:: mindspore.load_mindir(file_name)

    加载MindIR文件。

    .. note::
        这个接口在2.5版本废弃，并且会在未来版本移除。

    参数：
        - **file_name** (str) - MindIR文件的全路径名。

    返回：
        ModelProto，一个MindIR proto 对象。

    异常：
        - **ValueError** - 文件不存在或文件名格式不对。

