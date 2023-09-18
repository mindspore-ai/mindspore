mindspore.save_mindir
=======================================

.. py:function:: mindspore.save_mindir(model, file_name)

    保存MindIR文件。

    参数：
        - **model** (ModelProto) - MindIR model。
        - **file_name** (str) - MindIR文件的全路径名。

    异常：
        - **TypeError** - 参数 `model` 不是ModelProto对象。
        - **ValueError** - 文件路径不存在或文件名格式不对。
