mindspore.convert_model
======================

.. py:function:: mindspore.convert_model(mindir_file, convert_file, file_format)

    将MindIR模型转化为其他格式的模型文件。当前版本仅支持转化成ONNX模型。

    .. note::
        这是一个实验特性，未来API可能会发生的变化。

    **参数：**

    - **mindir_file** (str) - MindIR模型文件名称。
    - **convert_file** (str) - 转化后的模型文件名称。
    - **file_format** (str) - 需要转化的文件格式，当前版本仅支持"ONNX"。

    **异常：**

    - **TypeError** - `mindir_file` 参数不是str类型。
    - **TypeError** - `convert_file` 参数不是str类型。
    - **ValueError** - `file_format` 参数的值不是"ONNX"。
