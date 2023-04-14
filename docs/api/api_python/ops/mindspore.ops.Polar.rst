mindspore.ops.Polar
====================

.. py:class:: mindspore.ops.Polar

    将极坐标转化为笛卡尔坐标。

    更多细节请参考 :func:`mindspore.ops.polar`。

    输入：
        - **abs** (Tensor) - 极径。Tensor的shape为 :math:`(N,*)` ，其中 :math:`N` 为输入Tensor的批量大小， :math:`*` 为任意数量的额外维度。其数据类型须为：float32、float64。
        - **angle** (Tensor) - 极角。其shape与数据类型与 `abs` 一致。

    输出：
        Tensor，其shape与 `abs` 一致。
