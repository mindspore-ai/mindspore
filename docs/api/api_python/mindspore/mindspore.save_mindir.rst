mindspore.save_mindir
=======================================

.. py:function:: mindspore.save_mindir(model, file_name)

    保存MindIR文件。


    参数：
        - **file_name** (str) - MindIR文件的全路径名。
        - **model** (mindir_model) - MindIR model 。

    返回：
        None。

    异常：
        - **ValueError** - MindIR文件名不存在或 `file_name` 不是string类或model不是mindir model。

