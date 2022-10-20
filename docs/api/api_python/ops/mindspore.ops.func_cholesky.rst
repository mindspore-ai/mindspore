mindspore.ops.cholesky
======================

.. py:function:: mindspore.ops.cholesky(input_x, upper=False)

    计算对称正定矩阵 :math:`A` 或对称正定矩阵批次的Cholesky分解。

    如果 `upper` 为True，则返回的矩阵 :math:`U` 为上三角矩阵，分解形式为：

    .. math::
        A = U^TU

    如果 `upper` 为False，则返回的矩阵 :math:`L` 为下三角矩阵，分解形式为：
   
    .. math::
        A = LL^T 

    参数：
        - **input_x** (Tensor) - shape大小为 :math:`(*, N, N)` ，其中 :math:`*` 是零个或多个由对称正定矩阵组成的批处理维，数据类型为float32或float64。
        - **upper** (bool) - 是否返回上三角矩阵还是下三角矩阵的标志。默认值：False。

    返回：
        Tensor，shape和数据类型与 `input_x` 相同。

    异常：
        - **TypeError** - 如果 `upper` 不是bool。
        - **TypeError** - 如果 `input_x` 的数据类型既不是float32，也不是float64。
        - **TypeError** - 如果 `input_x` 不是Tensor。
        - **ValueError** - 如果 `input_x` 不是批处理方。
        - **ValueError** - 如果 `input_x` 不是对称正定矩阵。
