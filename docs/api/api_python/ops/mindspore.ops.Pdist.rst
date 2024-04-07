mindspore.ops.Pdist
===================

.. py:class:: mindspore.ops.Pdist(p=2.0)

    计算输入中每对行向量之间的p-范数距离。

    更多细节请参考 :func:`mindspore.ops.pdist`。

    .. note::
        Pdist算子由于涉及求幂次方运算，在float16类型输入场景下，很容易会因为数值溢出而产生inf/nan的计算结果，建议采用float32类型输入。

    参数：
        - **p** (float，可选) - 范数距离的阶， :math:`p∈[0, ∞)`。默认值： ``2.0`` 。

    输入：
        - **x** (Tensor) - 输入Tensor。类型：float16、float32或float64。

    输出：
        Tensor，类型与 `x` 一致。
