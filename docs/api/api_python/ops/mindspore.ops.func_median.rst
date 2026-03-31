mindspore.ops.median
====================

.. py:function:: mindspore.ops.median(input, axis=-1, keepdims=False)

    返回tensor在指定轴上的中位数及其索引。

    .. warning::
        - 如果 `input` 的中值不唯一，则 `indices` 不一定对应第一个出现的中值。该接口的具体实现方式和后端类型相关，CPU和GPU的返回值可能不相同。
        - 2.9.0（不含）之后版本中，默认 `axis` 将从 ``-1`` 改为 ``None``。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (int，可选) - 指定计算的轴。默认 ``-1`` 。
        - **keepdims** (bool，可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        两个tensor组成的tuple(median, median_indices)
