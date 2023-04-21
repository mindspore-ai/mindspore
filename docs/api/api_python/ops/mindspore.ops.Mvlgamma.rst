mindspore.ops.Mvlgamma
=======================

.. py:class:: mindspore.ops.Mvlgamma(p)

    逐元素计算 `p` 维多元对数伽马函数值。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多细节请参考 :func:`mindspore.ops.mvlgamma` 。

    参数：
        - **p** (int) - 进行计算的维度，必须大于等于1。

    输入：
        - **x** (Tensor) - 多元对数伽马函数的输入Tensor，支持数据类型为float32和float64。其shape为 :math:`(N,*)` ，其中 :math:`*` 为任意数量的额外维度。 `x` 中每个元素的值必须大于 :math:`(p - 1) / 2` 。

    输出：
        Tensor。shape和类型与 `x` 一致。
