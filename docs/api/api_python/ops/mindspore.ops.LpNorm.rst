mindspore.ops.LpNorm
=====================

.. py:class:: mindspore.ops.LpNorm(axis, p=2, keep_dims=False, epsilon=1e-12)

    返回输入Tensor的矩阵范数或向量范数。

    .. math::
        output = sum(abs(input)**p)**(1/p)

    有关更多详细信息，请参阅： :func:`mindspore.ops.norm` 。
