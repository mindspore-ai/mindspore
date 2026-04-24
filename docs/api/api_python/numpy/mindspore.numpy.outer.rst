mindspore.numpy.outer
=====================

.. py:function:: mindspore.numpy.outer(a, b)

    计算两个向量的外积。

    指定两个向量， ``a = [a0, a1, ..., aM]`` 和 ``b = [b0, b1, ..., bN]`` ，外积为：

    ``[[a0*b0  a0*b1 ... a0*bN ]``

    ``[a1*b0    .              ]``

    ``[ ...          .         ]``

    ``[aM*b0            aM*bN ]]``

    .. note::
        - 不支持NumPy参数 `out` 。
        - 在 GPU 上，支持的数据类型为 np.float16 。
        - 在 CPU 上，支持的数据类型为 np.float32 和 np.float64。

    参数：
        - **a** (Tensor) - 第一个输入向量。如果不是一维的，则将其展平。
        - **b** (Tensor) - 第二个输入向量。如果不是一维的，则将其展平。

    返回：
        Tensor或标量， ``out[i, j] = a[i] * b[j]``  。

    异常：
        - **TypeError** - 如果输入不是Tensor。