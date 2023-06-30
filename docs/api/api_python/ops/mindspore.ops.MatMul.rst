mindspore.ops.MatMul
=====================

.. py:class:: mindspore.ops.MatMul(transpose_a=False, transpose_b=False)

    将矩阵 `a` 和矩阵 `b` 相乘。

    .. math::

        (Output)_{i j}=\sum_{k=1}^{p} a_{i k} b_{k j}=a_{i 1} b_{1 j}+a_{i 2} b_{2 j}+\cdots+a_{i p} b_{p j}, p\in N

    其中， :math:`i,j` 表示输出的第i行和第j列元素。

    .. note::
        对于 :math:`N * M` 不能被16整除的情况下，在Ascend环境上性能会比较差。输入Tensor的数据类型必须一致。

    参数：
        - **transpose_a** (bool) - 如果为 ``True`` ，则在相乘之前转置 `a`。默认值： ``False`` 。
        - **transpose_b** (bool) - 如果为 ``True`` ，则在相乘之前转置 `b`。默认值： ``False`` 。

    输入：
        - **a** (Tensor) - 要相乘的第一个Tensor。如果 `transpose_a` 为 ``False`` ，则该Tensor的shape为 :math:`(N, C)` ；否则，该Tensor的shape为 :math:`(C, N)` 。
        - **b** (Tensor) - 要相乘的第二个Tensor。如果 `transpose_b` 为 ``False`` ，则该Tensor的shape为 :math:`(C, M)` ；否则，该Tensor的shape为 :math:`(M, C)` 。

    输出：
        Tensor，输出Tensor的shape为 :math:`(N, M)` 。

    异常：
        - **TypeError** - `transpose_a` 或 `transpose_b` 不是bool。
        - **TypeError** - `a` 的dtype和 `b` 的dtype不一致。
        - **ValueError** - 矩阵 `a` 的列不等于矩阵 `b` 的行。
        - **ValueError** - `a` 或 `b` 的维度不等于2。
