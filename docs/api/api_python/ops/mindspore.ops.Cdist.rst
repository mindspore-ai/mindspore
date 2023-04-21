mindspore.ops.Cdist
===================

.. py:class:: mindspore.ops.Cdist(p=2.0)

    计算两个Tensor的p-norm距离。

    更多参考详见 :func:`mindspore.ops.cdist`。

    参数：
        - **p** (float，可选) - 计算向量对p-norm距离的P值，P∈[0，∞]。默认值： ``2.0`` 。

    输入：
        - **input_x** (Tensor) - 输入Tensor，shape为 :math:`(B, P, M)` ， :math:`B` 维度为0时该维度被忽略，shape为 :math:`(P, M)` 。
        - **input_y** (Tensor) - 输入Tensor，shape为 :math:`(B, R, M)` ，与 `input_x` 的数据类型一致。

    输出：
        Tensor，p-norm距离，数据类型与 `input_x` 一致，shape为 :math:`(B, P, R)`。
