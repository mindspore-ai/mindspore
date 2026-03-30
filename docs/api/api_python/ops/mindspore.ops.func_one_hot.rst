mindspore.ops.one_hot
=====================

.. py:function:: mindspore.ops.one_hot(indices, depth, on_value=1, off_value=0, axis=-1)

    生成一个新的tensor，索引 `indices` 表示的位置上取值 `on_value` ，其他所有位置上取值 `off_value` 。

    .. warning::
        从2.9.0（不含）之后版本开始，接口参数将调整为 `indices, num_classes`，并移除 `depth`、`on_value`、`off_value` 和 `axis`。

    .. note::
        如果 `indices` 的秩为 `n` ，则输出Tensor的秩为 `n+1` 。新轴在 `axis` 处创建。当执行设备是 Ascend 时，如果 `on_value` 为int64类型，则 `indices` 也必须为int64类型，且 `on_value` 和 `off_value` 的取值只能是1和0。

    参数：
        - **indices** (Tensor) - 输入索引。
        - **depth** (int) - one-hot深度。
        - **on_value** (Union[Tensor, int, float]，可选) - 填充索引位置上的值。默认 ``1``。
        - **off_value** (Union[Tensor, int, float]，可选) - 填充非索引位置上的值。默认 ``0``。
        - **axis** (int，可选) - 指定计算轴。默认 ``-1`` 。

    返回：
        Tensor
