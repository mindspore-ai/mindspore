mindspore.ops.cholesky
======================

.. py:function:: mindspore.ops.cholesky(input_x, upper=False)

    计算给定对称正定矩阵的Cholesky分解。

    .. warning::
        从2.9.0（不含）之后版本开始，参数 `input_x` 将重命名为 `input`。

    如果 `upper` 为True，则返回的矩阵 :math:`U` 为上三角矩阵，分解形式为：

    .. math::
        A = U^TU

    如果 `upper` 为False，则返回的矩阵 :math:`L` 为下三角矩阵，分解形式为：

    .. math::
        A = LL^T

    参数：
        - **input_x** (Tensor) - shape大小为 :math:`(*, N, N)` 的输入tensor，公式中的 :math:`A` 。
        - **upper** (bool，可选) - 是否返回上三角矩阵，默认 ``False`` 。

    返回：
        Tensor
