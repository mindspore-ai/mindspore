mindspore.numpy.multi_dot
=========================

.. py:function:: mindspore.numpy.multi_dot(arrays)

    计算两个或更多个数组的点积，同时自动选择最快的计算顺序。

    multi_dot将链式调用numpy.dot，并使用矩阵的最佳括号化。
    更多信息，请参考 `维基百科 <https://en.wikipedia.org/wiki/Matrix_chain_multiplication>`_。
    根据矩阵的shape，这可以大大加快乘法速度。 如果第一个参数是一维的，则将其视为行向量。
    如果最后一个参数是一维的，则将其视为列向量。 其他参数必须是二维的。

    .. note::
        不支持NumPy参数 `out` 。

    参数：
        - **arrays** (类数组的序列) - 如果第一个参数是一维的，则将其视为行向量。如果最后一个参数是一维的，则将其视为列向量。其他参数必须是二维的。

    返回：
        Tensor，所提供数组的点积。

    异常：
        - **ValueError** - 数组不是二维的。