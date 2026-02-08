mindspore.ops.mv
=================

.. py:function:: mindspore.ops.mv(mat, vec)

    实现矩阵 `mat` 和向量 `vec` 相乘。

    如果 `mat` 的shape为 :math:`(N, M)` ， `vec` 的shape为 :math:`M` ，则输出为 :math:`N` 的一维tensor。

    参数：
        - **mat** (Tensor) - 输入矩阵。
        - **vec** (Tensor) - 输入向量。

    返回：
        Tensor
