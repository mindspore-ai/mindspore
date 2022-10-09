mindspore.ops.MatrixInverse
============================

.. py:class:: mindspore.ops.MatrixInverse(adjoint=False)

    计算输入矩阵的逆矩阵。如果输入矩阵不可逆，将产生错误或者返回一个未知结果。

    .. note::
        参数 `adjoint` 目前只支持False，因为目前该算子不支持复数。

    参数：
        - **adjoint** (bool) - 指定是否支持复数，False表示为不支持复数。默认：False。

    输入：
        - **x** (Tensor) - 输入需计算的矩阵，至少为二维矩阵，且最后两个维度大小相同，数据类型为float32、float64。

    输出：
        Tensor，数据类型和shape与输入 `x` 相同。

    异常：
        - **TypeError** - `adjoint` 不是bool。
        - **TypeError** - `x` 的数据类型既不是float32，也不是float64。
        - **ValueError** - `x` 最后两个维度大小不同。
        - **ValueError** - `x` 低于二维。
