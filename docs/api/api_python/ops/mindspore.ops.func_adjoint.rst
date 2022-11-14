mindspore.ops.adjoint
======================

.. py:function:: mindspore.ops.adjoint(x)

    计算张量的共轭，并转置最后两个维度。

    参数：
        - **x** (Tensor) - 参与计算的tensor。

    返回：
        Tensor，和 `x` 具有相同的dtype和shape。

    异常：
        - **TypeError**：`x` 不是tensor。