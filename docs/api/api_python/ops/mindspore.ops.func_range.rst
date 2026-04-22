mindspore.ops.range
====================

.. py:function:: mindspore.ops.range(start, end, step, maxlen=1000000)

    返回一个在 [ `start`, `end` ) 区间内，步长为 `step` 的tensor。

    .. note::
        - 三个输入的数据类型必须全为整数或全为浮点数。
        - 当输入为tensor时，tensor中仅可包含一个元素，数据类型为数值型。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：

        - `start` 将默认为 ``0`` ，`step` 将默认为 ``1`` 。
        - 参数 `maxlen` 将被删除。

        变更后的接口签名为 ``mindspore.ops.range(start=0, end, step=1)`` 。

    参数：
        - **start** (Union[Number, Tensor]) - 区间的起始值。
        - **end** (Union[Number, Tensor]) - 区间的末尾数。
        - **step** (Union[Number, Tensor]) - 值的间隔。
        - **maxlen** (int，可选) - 该算子将会被分配能够存储 `maxlen` 个数据的内存。该参数必须为正数，默认 ``1000000`` 。如果输出的数量超过 `maxlen` ，将会引起运行时错误。

    返回：
        Tensor
