mindspore.set_dump
==================

.. py:class:: mindspore.set_dump(target, enabled=True)

    启用或者禁用target及其子节点的Dump数据功能。

    target为 `Cell <https://www.mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell>`_ 或 `Primitive <https://www.mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.Primitive.html#mindspore.ops.Primitive>`_ 的实例。请注意，此API仅在开启异步Dump功能且Dump配置文件中的 `dump_mode` 字段为"2"时生效。有关详细信息，请参阅 `Dump功能文档 <https://mindspore.cn/docs/programming_guide/zh-CN/master/dump_in_graph_mode.html>`_ 。默认状态下，Cell和Primitive实例不使能Dump数据功能。

    .. Warning::
        此API还在实验阶段，后续可能修改或删除。

    .. Note::
        - 此API只在Ascend后端的图模式有效。
        - 当target是一个Cell且enabled设置为True时，Cell实例及其子Cell实例的Primitive将递归启用Dump。如果算子不是Cell实例的成员，则不会为该算子启用Dump（例如，在construct方法中直接使用的 `functional 算子 <https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.ops.html#functional>`_ ）。要使此API生效，请在Cell的__init__方法中使用self.some_op = SomeOp()的写法。
        - 使用set_dump(Cell, True)后，Cell正向计算中的算子会被Dump，大多数反向计算（梯度运算产生的计算）不会被Dump。然而，由于图的优化，一些反向计算的数据仍然会被Dump。可以忽略文件名中包含“Gradients”的反向计算数据。
        - 此API只支持训练开始前调用。如果在训练过程中调用这个API，可能不会有效果。
        - 对于 `nn.SoftMaxCrossEntropyWithLogits 层 <https://www.mindspore.cn/docs/api/zh-CN/master/api_python/nn/mindspore.nn.SoftmaxCrossEntropyWithLogits.html#mindspore.nn.SoftmaxCrossEntropyWithLogits>`_ ，正向计算和反向计算使用同一组算子。因此，只能看到反向计算中的Dump数据。请注意，当使用sparse=True和reduce=“mean”初始化时，nn.SoftmaxCrossEntropyWithLogits层也将在内部使用这些算子。

    **参数：**

    - **target** (Union[Cell, Primitive]) - 要设置Dump标志的Cell或Primitive的实例。
    - **enabled** (bool) - True表示启用Dump，False表示禁用Dump，默认值: True。