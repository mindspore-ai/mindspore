mindspore.ops.csr_add
=================================

.. py:function:: mindspore.ops.csr_add(a: CSRTensor, b: CSRTensor, alpha: Tensor, beta: Tensor)

    :math:`a` 和 :math:`b` 是CSRTensor，:math:`alpha` 和 :math:`beta` 是Tensor。

    .. math::

        out = alpha * a + beta * b

    .. note::
        用户需要确保输入的稀疏算子的合法性。否则，算子将返回错误结果。
        例如，当同一个位置有多个元素时，算子可能会返回错误结果或执行失败。

    参数：
        - **a** (CSRTensor) - 稀疏的 CSRTensor。
        - **b** (CSRTensor) - 稀疏的 CSRTensor。
        - **alpha** (Tensor) - 稠密张量，shape必须可以广播到 `a`。
        - **beta** (Tensor) - 稠密张量，shape必须可以广播到 `b`。

    返回：
        返回一个包含以下数据的CSRTensor。

        - **indptr** - 指示每行中非零值的起始点和结束点。
        - **indices** - 输入中所有非零值的列位置。
        - **values** - 稠密张量的非零值。
        - **shape** - csr_tensor 的形状。