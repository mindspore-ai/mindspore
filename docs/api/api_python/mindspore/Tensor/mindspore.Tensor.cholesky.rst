mindspore.Tensor.cholesky
=========================

.. py:method:: mindspore.Tensor.cholesky(upper=False)

   计算对称正定矩阵 :math:`A` 或对称正定矩阵批次的Cholesky分解。

    如果 `upper` 为True，则返回的矩阵 :math:`U` 为上三角矩阵，分解形式为：

    .. math::
        A = U^TU

    如果 `upper` 为False，则返回的矩阵 :math:`L` 为下三角矩阵，分解形式为：

    .. math::
        A = LL^T

    参数：
        - **upper** (bool) - 返回上三角矩阵/下三角矩阵的标志。True为上三角矩阵，False为下三角矩阵。默认值：False。

    返回：
        Tensor，shape和数据类型与输入相同。

    异常：
        - **TypeError** - 如果 `upper` 不是bool。
        - **TypeError** - 当前Tensor的数据类型既不是float32，也不是float64。
        - **ValueError** - 当前Tensor不是批处理方。
        - **ValueError** - 如果Tensor不是对称正定矩阵。
