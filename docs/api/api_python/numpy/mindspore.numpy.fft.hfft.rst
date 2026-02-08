mindspore.numpy.fft.hfft
========================

.. py:function:: mindspore.numpy.fft.hfft(a, n=None, axis=-1, norm=None)

    计算 Hermitian 对称的 `a` 信号的一维离散傅里叶变换。

    请参阅 :func:`mindspore.ops.hfft` 以获取更多详细信息。 区别在于 `a` 对应于 `input` ， `axis` 对应于 `dim` 。

    参数：
        - **a** (Tensor) - 输入Tensor。 支持的dtype： - Ascend/CPU:int16, int32, int64, float16, float32, float64, complex64, complex128。
        - **n** (tuple[int], 可选) - 结果中 `axis` 轴变换为的长度。 如果给定，输入将在计算 `hfft` 之前进行零填充或截断为长度 `n` 。 默认值： ``None`` ，表示不处理 `a` 。
        - **axis** (tuple[int], 可选) - 计算 `hfft` 所沿的轴。 默认值： ``-1`` ，表示在 `a` 最后一维上计算。
        - **norm** (str, 可选) - 归一化模式。 默认值： ``None`` ，表示 ``"backward"`` 。 三种模式的定义如下， ``"backward"`` (无归一化)， ``"forward"`` (按 :math:`1/n` 归一化)， ``"ortho"`` (按 :math:`1/\sqrt{n}` 归一化)。

    返回：
        Tensor， `hfft()` 函数的结果。
        如果给定 `n` ，则 `result.shape[axis]` 为 :math:`(n - 1) * 2` ，否则为 :math:`(a.shape[axis] - 1) * 2` 。
        当 `a` 是 int16、int32、int64、float16、float32、complex64 类型时，返回值类型为 complex64。
        当 `a` 是 float64 或 complex128 类型时，返回值类型为 complex128。