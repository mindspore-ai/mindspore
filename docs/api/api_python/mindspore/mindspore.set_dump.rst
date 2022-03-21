mindspore.set_dump
==================

.. py:class:: mindspore.set_dump(target, enabled=True)

    启用或者禁用target及其子节点的Dump数据功能。

    target为 `Cell <https://www.mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell>`_ 或 `Primitive <https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Primitive.html#mindspore.ops.Primitive>`_ 的实例。请注意，此API仅在开启异步Dump功能且Dump配置文件中的 `dump_mode` 字段为"2"时生效。有关详细信息，请参阅 `Dump功能文档 <https://mindspore.cn/docs/programming_guide/zh-CN/master/dump_in_graph_mode.html>`_ 。默认状态下，Cell和Primitive实例不使能Dump数据功能。

    .. Warning::
        此类中的所有API均为实验版本，将来可能更改或者删除。

    .. Note::
        - 此API只在Ascend后端的图模式有效。
        - 此API只支持训练开始前调用。如果在训练过程中调用这个API，可能不会有效果。
        - 使用set_dump(Cell, True)后，Cell正向计算和反向计算（梯度运算产生的计算）中的算子会被Dump。
        - 对于 `nn.SoftMaxCrossEntropyWithLogits 层 <https://www.mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.SoftmaxCrossEntropyWithLogits.html#mindspore.nn.SoftmaxCrossEntropyWithLogits>`_ ，正向计算和反向计算使用同一组算子。因此，只能看到反向计算中的Dump数据。请注意，当使用sparse=True和reduce=“mean”初始化时，nn.SoftmaxCrossEntropyWithLogits层也将在内部使用这些算子。

    **参数：**

    - **target** (Union[Cell, Primitive]) - 要设置Dump标志的Cell或Primitive的实例。
    - **enabled** (bool) - True表示启用Dump，False表示禁用Dump，默认值: True。