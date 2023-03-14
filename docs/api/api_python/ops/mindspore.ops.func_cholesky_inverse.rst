mindspore.ops.cholesky_inverse
==============================

.. py:function:: mindspore.ops.cholesky_inverse(input_x, upper=False)

    使用Cholesky分解计算对称正定矩阵的逆矩阵。

    如果 `upper` 为True，则返回的矩阵 :math:`U` 为上三角矩阵，输出的结果：
   
    .. math::
        inv = (U^{T}U)^{-1}

    如果 `upper` 为False，则返回的矩阵 :math:`U` 为下三角矩阵，输出的结果：

    .. math::
        inv = (UU^{T})^{-1}

    .. note::
       输入Tensor必须是一个由Cholesky分解得到的上三角矩阵或者下三角矩阵。    

    参数：
        - **input_x** (Tensor) - 输入Tensor，其rank为2，数据类型为float32或float64。
        - **upper** (bool) - 返回上三角矩阵还是下三角矩阵的标志。默认值：False。

    返回：
        Tensor，shape和数据类型与 `input_x` 相同。

    异常：
        - **TypeError** - 如果 `input_x` 不是Tensor。
        - **TypeError** - 如果 `input_x` 的数据类型既不是float32，也不是float64。
        - **ValueError** - 如果 `input_x` 的维度不等于2。
