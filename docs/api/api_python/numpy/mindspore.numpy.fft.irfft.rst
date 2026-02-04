mindspore.numpy.fft.irfft
=========================

.. py:function:: mindspore.numpy.fft.irfft(a, n=None, axis=-1, norm=None)

    计算 `rfft` 的逆变换。

    参见 :func:`mindspore.ops.irfft` 获取更多详情。不同之处在于 `a` 对应 `input` , `axis` 对应 `dim` 。

    参数：
        - **a** (Tensor) - 输入Tensor。
        - **n** (int, 可选) - 结果 `axis` 轴变换为的长度。 如果给定，在计算 `irfft` 之前， `axis` 轴的大小将进行零填充或截断为 `n` 。 如果没有给定 `n` ，长度为 :math:`2*(a.shape[axis]-1)` 。 默认值： ``None`` 。
        - **axis** - (int, 可选) - 计算 `irfft` 的轴。 默认: ``-1`` ，表示使用 `a` 的最后一个轴。
        - **norm** (str, 可选) - 归一化模式。 默认值： ``None`` ，表示 ``"backward"`` 。 三种模式的定义如下， ``"backward"`` (无归一化)， ``"forward"`` (按 :math:`1*n` 归一化)， ``"ortho"`` (按 :math:`1*\sqrt{n}` 归一化)。

    返回：
        Tensor， `irfft()` 函数的结果。