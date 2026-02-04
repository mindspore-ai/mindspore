mindspore.numpy.fft.ihfft
=========================

.. py:function:: mindspore.numpy.fft.ihfft(a, n=None, axis=-1, norm=None)

    计算 `ihfft` 的逆变换。

    请参阅 :func:`mindspore.ops.ihfft` 以获取更多详细信息。 区别在于 `a` 对应于 `input` ， `axis` 对应于 `dim` 。

    参数：
        - **a** (Tensor) - 输入Tensor。 支持的dtype： - Ascend/CPU:int16, int32, int64, float16, float32, float64, complex64, complex128。
        - **n** (tuple[int], 可选) - 结果中 `axis` 轴变换为的长度。 如果给定，输入将在计算 `ihfft` 之前进行零填充或截断为长度 `n` 。 默认值： ``None`` ，表示不处理 `a` 。
        - **axis** (tuple[int], 可选) - 计算 `ihfft` 所沿的轴。 默认值： ``-1`` ，表示在 `a` 最后一维上计算。
        - **norm** (str, 可选) - 归一化模式。 默认值： ``None`` ，表示 ``"backward"`` 。 三种模式的定义如下， ``"backward"`` (无归一化)， ``"forward"`` (按 :math:`1*n` 归一化)， ``"ortho"`` (按 :math:`1*\sqrt{n}` 归一化)。

    返回：
        Tensor， `ihfft()` 函数的结果。
        如果给定 `n` ，则 `result.shape[axis]` 为 :math:`n // 2 + 1` ，否则为 :math:`a.shape[axis] // 2 + 1` 。
        当 `a` 是 int16、int32、int64、float16、float32、complex64 类型时，返回值类型为 complex64。
        当 `a` 是 float64 或 complex128 类型时，返回值类型为 complex128。

    异常：
        - **TypeError** - 如果 `a` 的类型不是 Tensor。
        - **TypeError** - 如果 `a` 的数据类型不是以下之一: int32, int64, float32, float64。
        - **TypeError** - 如果 `n` 或 `axis` 的类型不是 int。
        - **ValueError** - 如果 `axis` 不在 "[ `-a.ndim` , `a.ndim` )" 范围内。
        - **ValueError** - 如果 `n` 小于 1。
        - **ValueError** - 如果 `norm` 不是 `"backward"` , `"forward"` 或 `"ortho"` 之一。