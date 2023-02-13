mindspore.ops.adjoint
======================

.. py:function:: mindspore.ops.adjoint(x)

    计算Tensor的共轭，并转置最后两个维度。

    参数：
        - **x** (Tensor) - 参与计算的Tensor。

    返回：
        Tensor，和 `x` 具有相同的dtype和shape。

    异常：
        - **TypeError**：`x` 不是Tensor。