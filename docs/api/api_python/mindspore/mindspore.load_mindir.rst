mindspore.load_mindir
=======================================

.. py:function:: mindspore.load_mindir(file_name)

    加载MindIR文件。

    返回一个 `mindir_model` 的对象。

    参数：
        - **file_name** (str) - MindIR文件的全路径名。

    返回：
        mindir_model。

    异常：
        - **ValueError** - MindIR文件名不存在或 `file_name` 不是string类型。
        - **RuntimeError** - 解析MindIR文件失败。

