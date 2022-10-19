mindspore.Tensor.cholesky_inverse
=================================

.. py:method:: mindspore.Tensor.cholesky_inverse(upper=False)

    计算对称正定矩阵的逆矩阵。

    如果 `upper` 为False，则返回的矩阵 :math:`U` 为下三角矩阵，输出的结果：

    .. math::
        inv = (UU^{T})^{-1}

    如果 `upper` 为True，则返回的矩阵 :math:`U` 为上三角矩阵，输出的结果：

    .. math::
        inv = (U^{T}U)^{-1}

    .. note::
       当前Tensor必须是一个上三角矩阵或者下三角矩阵。

    参数：
        - **upper** (bool) - 返回上三角矩阵/下三角矩阵的标志。True为上三角矩阵，False为下三角矩阵。默认值：False。

    返回：
        Tensor，shape和数据类型与输入相同。

    异常：
        - **TypeError** - 当前Tensor的数据类型既不是float32，也不是float64。
        - **ValueError** - 当前Tensor的维度不等于2。
