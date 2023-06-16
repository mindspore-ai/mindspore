mindspore.ops.LinSpace
======================

.. py:class:: mindspore.ops.LinSpace

    返回一个在区间 `start` 和 `stop` （包括 `start` 和 `stop` ）内均匀分布的，包含 `num` 个值的一维Tensor。

    更多参考详见 :func:`mindspore.ops.linspace`。

    输入：
        - **start** (Tensor) - 区间的起始值。零维Tensor，数据类型为float32或float64。
        - **stop** (Tensor) - 区间的末尾值。零维Tensor，数据类型为float32或float64。
        - **num** (int) - 间隔中的包含的数值数量，包括区间端点。支持的数据类型：int32、int64。

    输出：
        Tensor，与 `start` 的shape和数据类型相同。
