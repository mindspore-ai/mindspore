mindspore.onnx.export
=====================

.. py:function:: mindspore.onnx.export(net, *inputs, file_name, input_names=None, output_names=None, export_params=True, keep_initializers_as_inputs=False, dynamic_axes=None)

    将MindSpore网络模型导出为ONNX文件。

    .. note::
        - 支持模型大小超过2GB导出ONNX，当检测到模型大小超过2GB时，模型的参数会保存在额外的二进制文件中，与ONNX文件保存在同级目录。
        - 当 `file_name` 没有后缀时，系统会自动添加后缀 `.onnx` 。

    参数：
        - **net** (Union[Cell, function]) - MindSpore网络结构。
        - **inputs** (Union[Tensor, list, tuple, Number, bool]) - 网络的输入，如果网络有多个输入，需要一同传入。
        - **file_name** (str) - 导出模型的文件名称。
        - **input_names** (list, 可选) - 按顺序为图的输入节点修改名称。默认值： ``None`` 。
        - **output_names** (list, 可选) - 按顺序为图的输出节点修改名称。默认值： ``None`` 。
        - **export_params** (bool, 可选) - 如果设置为False，参数（权重）将不会被导出到ONNX模型中，而作为模型的输入节点。默认值： ``True`` 。
        - **keep_initializers_as_inputs** (bool, 可选) - 如果设置为True，所有的初始化值（模型参数、模型权重）也将被添加为图的输入，当运行导出的ONNX模型时，如果想替换某一或所有权重，请设置为True。默认值：``False`` 。
        - **dynamic_axes** (dict[str, dict[int, str]], 可选) - 将输入节点张量的轴指定为动态的（运行时）。默认值： ``None`` 。

          - 参数格式为{"输入节点": {轴索引: "轴名称"}}，例如：{"input1": {0:"batch_size", 1: "seq_len"}, "input2": {0:"batch_size"}}。
          - 默认情况下，导出的模型的所有输入张量的形状与 `inputs` 中给定的形状完全匹配。

    异常：
        - **ValueError** - 参数 `net` 的类型不为 :class:`mindspore.nn.Cell` 对象。
        - **ValueError** - 参数 `input_names` 的类型不为list。
        - **ValueError** - 参数 `output_names` 的类型不为list。
        - **ValueError** - 参数 `dynamic_axes` 的类型不为dict。
