mindspore.numpy.inner
=====================

.. py:function:: mindspore.numpy.inner(a, b)

    返回两个Tensor的内积。

    对于1-D的Tensor，这是向量的普通内积(不包含复共轭)。对于更高维的Tensor，这是在最后一个轴上的求和积。

    .. note::
        - 不支持NumPy参数 `out`。
        - 在GPU上，支持的数据类型有np.float16和np.float32。
        - 在CPU上，支持的数据类型有np.float16，np.float32和np.float64。

    参数：
        - **a** (Tensor) - 输入Tensor。 如果 `a` 和 `b` 都不是标量，它们的最后一个维度必须匹配。
        - **b** (Tensor) - 输入Tensor。 如果 `a` 和 `b` 都不是标量，它们的最后一个维度必须匹配。

    返回：
        Tensor或标量。

    异常：
        - **ValueError** - 如果 ``x1.shape[-1] != x2.shape[-1]`` 。