mindspore.numpy.dot
===================

.. py:function:: mindspore.numpy.dot(a, b)

    返回两个数组的点积。

    具体来说，如果 `a` 和 `b` 都是1-D数组，它是向量的内积(没有复数共轭)。 如果 `a` 和 `b` 都是2-D数组，它是矩阵乘法。
    如果 `a` 或 `b` 是0-D的(标量)，则等同于乘法。 如果 `a` 为N-D数组且 `b` 为1-D数组，它在 `a` 的最后一个轴上与 `b` 计算求和乘积。
    如果 `a` 是N-D数组且 `b` 是M-D数组(其中M>=2)，它在 `a` 的最后一个轴上与 `b` 的倒数第二个轴上计算求和乘积： ``dot(a, b)[i,j,k,m] = sum(a[i,j, :] * b[k, :, m])`` 。

    .. note::
        - 不支持NumPy的 `out` 参数。
        - 在 GPU 上，支持的数据类型为 np.float16、np.float32 和 np.float64。
        - 在 CPU 上，支持的数据类型为 np.float16、np.float32 和 np.float64。

    参数：
        - **a** (Tensor) - 输入Tensor。
        - **b** (Tensor) - 输入Tensor。

    返回：
        Tensor或标量， `a` 和 `b` 的点积。如果 `a` 和 `b` 都是标量或都是1-D数组，则返回一个标量，否则返回一个数组。

    异常：
        - **ValueError** - 如果 `a` 的最后一个维度的大小与 `b` 的倒数第二个维度的大小不同。