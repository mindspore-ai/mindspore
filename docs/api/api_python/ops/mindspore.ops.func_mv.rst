mindspore.ops.mv
=================

.. py:function:: mindspore.ops.mv(mat, vec)

    实现矩阵 `mat` 和向量 `vec` 相乘。

    如果 `mat` 是shape为 :math:`(N, M)` 的Tensor， `vec` 是长度为 :math:`M` 的一维Tensor，则输出是长度为 :math:`N` 的一维Tensor。

    参数：
        - **mat** (Tensor) - 输入矩阵，其shape为 :math:`(N, M)` 。
        - **vec** (Tensor) - 输入向量，其shape为 :math:`(M,)` 。

    返回：
        Tensor，其shape为 :math:`(N,)` 。

    异常：
        - **TypeError** - `mat` 或 `vec` 不是Tensor。
        - **ValueError** - `mat` 不是二维Tensor或 `vec` 不是一维Tensor。
