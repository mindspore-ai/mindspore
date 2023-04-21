mindspore.ops.CholeskyInverse
==============================

.. py:class:: mindspore.ops.CholeskyInverse(upper=False)

    使用Cholesky矩阵分解返回正定矩阵的逆矩阵。

    更多参考详见 :func:`mindspore.ops.cholesky_inverse`。

    参数：
        - **upper** (bool，可选) - 返回上三角矩阵还是下三角矩阵的标志。默认值： ``False`` 。

    输入：
        - **x** (Tensor) - 输入Tensor，其rank为2，数据类型为float32或float64。

    输出：
        Tensor，shape和数据类型与 `x` 相同。
