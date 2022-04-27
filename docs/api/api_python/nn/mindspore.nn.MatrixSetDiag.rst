mindspore.nn.MatrixSetDiag
===========================

.. py:class:: mindspore.nn.MatrixSetDiag

    将输入的对角矩阵的对角线值置换为输入的对角线值。

    假设 `x` 有 :math:`k+1` 个维度 :math:`[I,J,K,...,M,N]' ， `diagonal` 有 :math:`k` 个维度 :math:`[I, J, K, ..., min(M, N)]` ，则输出秩为 :math:`k+1` ，维度为 :math:`[I, J, K, ..., M, N]` 的Tensor，其中：

    .. math::
        output[i, j, k, ..., m, n] = diagnoal[i, j, k, ..., n]\ for\ m == n

    .. math::
        output[i, j, k, ..., m, n] = x[i, j, k, ..., m, n]\ for\ m != n

    **输入：**

    - **x** (Tensor) - 输入的对角矩阵。秩为k+1，k大于等于1。支持如下数据类型：float32、float16、int32、int8和uint8。
    - **diagonal** (Tensor) - 输入的对角线值。必须与输入 `x` 的shape相同。秩为k，k大于等于1。

    **输出：**

    Tensor，shape和数据类型与输入 `x` 相同。

    **异常：**

    - **TypeError** - `x` 或 `diagonal` 的数据类型不是float32、float16、int32、int8或uint8。
    - **ValueError** - `x` 的shape长度小于2。
    - **ValueError** - :math:`x_shape[-2] < x_shape[-1]` 且 :math:`x_shape[:-1] != diagonal_shape` 。
    - **ValueError** - :math:`x_shape[-2] >= x_shape[-1]` 且 :math:`x_shape[:-2] + x_shape[-1:] != diagonal_shape` 。 
    