mindspore.ops.Pdist
===================

.. py:class:: mindspore.ops.Pdist(p=2.0)

    计算输入中每对行向量之间的p-范数距离。

    更多细节请参考 :func:`mindspore.ops.pdist`。

    参数：
        - **p** (float，可选) - 范数距离的阶， :math:`p∈[0, ∞)`。默认值： ``2.0`` 。

    输入：
        - **x** (Tensor) - 输入Tensor `x` ，其shape为 :math:`(*B, N, M)`，其中 :math:`*B` 表示批处理大小，可以是多维度。类型：float16、float32或float64。

    输出：
        Tensor，类型与 `x` 一致。
