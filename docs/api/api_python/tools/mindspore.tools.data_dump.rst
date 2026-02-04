
.. py:function:: mindspore.tools.set_dump(target, enabled=True)

    启用或者禁用 `target` 及其子节点的Dump数据功能。

    `target` 为 :class:`mindspore.nn.Cell` 或 :class:`mindspore.ops.Primitive` 的实例。请注意，此API仅在 `ms_backend` 编译后端（请参考 `jit <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.jit.html>`_ 中backend参数），开启Dump功能且Dump配置文件中的 `dump_mode` 字段为 ``"2"`` 时生效。有关详细信息，请参阅 `Dump功能文档 <https://www.mindspore.cn/tutorials/zh-CN/master/debug/dump.html>`_ 。默认状态下， :class:`mindspore.nn.Cell` 和 :class:`mindspore.ops.Primitive` 实例不使能Dump数据功能。

    .. note::
        1. 此API只能在JIT编译中使用，要求目标设备为 "Ascend" 且使用 `ms_backend` 编译后端（请参考 `jit <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.jit.html>`_ 中backend参数），同时不支持融合算子。
        2. 此API只支持训练开始前调用。如果在训练过程中调用这个API，可能不会有效果。
        3. 使用 `set_dump(Cell, True)` 后，Cell正向计算和反向计算（梯度运算产生的计算）中的算子会被Dump。
        4. 对于 :class:`mindspore.nn.SoftmaxCrossEntropyWithLogits` 层，正向计算和反向计算使用同一组算子。因此，只能看到反向计算中的Dump数据。请注意，当使用 `sparse=True` 和 `reduce="mean"` 初始化时， :class:`mindspore.nn.SoftmaxCrossEntropyWithLogits` 层也将在内部使用这些算子。

    参数：
        - **target** (Union[Cell, Primitive]) - 要设置Dump标志的Cell或Primitive的实例。
        - **enabled** (bool，可选) - ``True`` 表示启用Dump， ``False`` 表示禁用Dump，默认值： ``True`` 。

    样例：

    .. note::
        运行此样例之前请设置环境变量 `MINDSPORE_DUMP_CONFIG` 到配置文件，并将配置文件中的 `dump_mode` 字段设置为2。
        详细信息请参阅 `Dump功能文档 <https://www.mindspore.cn/tutorials/zh-CN/master/debug/dump.html>`_ 。
