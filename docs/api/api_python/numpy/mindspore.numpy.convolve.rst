mindspore.numpy.convolve
========================

.. py:function:: mindspore.numpy.convolve(a, v, mode='full')

    返回两个一维序列的离散线性卷积。

    .. note::
        如果 `v` 比 `a` 长，计算前会交换这两个Tensor。

    参数：
        - **a** (Union[list, tuple, Tensor]) - 第一个输入的一维Tensor。
        - **v** (Union[list, tuple, Tensor]) - 第二个输入的一维Tensor。
        - **mode** (str, 可选) - 默认为 ``'full'`` ，在此 `mode` 下，卷积在每个重叠点计算，输出的shape为 :math:`(N + M - 1)` 。在卷积的端点，信号可能不完全重叠，可能会看到边界效应。如果 `mode` 为 `'same'`，则返回值的长度为 `max(M,N)` 。依旧可以观察到边界效应。如果 `mode` 为 `'valid'`，则返回值的长度为 `max(M, N) - min(M, N) + 1`。卷积仅在信号完全重叠的点给出。信号边界外的值无效。

    返回：
        Tensor， `a` 和 `v` 的离散线性卷积。

    异常：
        - **TypeError** - 如果输入的类型不符合上述规定。
        - **ValueError** - 如果 `a` 和 `v` 为空或维数错误。