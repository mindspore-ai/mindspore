mindspore.numpy.fft.irfftn
==========================

.. py:function:: mindspore.numpy.fft.irfftn(a, s=None, axes=None, norm=None)

    计算 `rfftn()` 的逆变换。

    请参阅 :func:`mindspore.ops.irfftn` 以获取更多详细信息。 区别在于 `a` 对应于 `input` ， `axes` 对应于 `dim` 。

    参数：
        - **a** (Tensor) - 输入Tensor。 支持的dtype： - Ascend/CPU:int16, int32, int64, float16, float32, float64, complex64, complex128。
        - **s** (tuple[int], 可选) - 结果中 `axes` 轴变换为的长度。 如果给定，输入将在计算 `irfftn` 之前进行零填充或截断为长度 `s` 。 默认值： ``None`` ，表示 axes[-1] 将零填充至 :math:`2*(a.shape[axes[-1]]-1)` 。
        - **axes** (tuple[int], 可选) - 计算 `irfftn` 所沿的轴。 默认值： ``(-2, -1)`` ，表示在 `a` 最后两个维度上计算。
        - **norm** (str, 可选) - 归一化模式。 默认值： ``None`` ，表示 ``"backward"`` 。 三种模式的定义如下， ``"backward"`` (无归一化)， ``"forward"`` (按 :math:`1*n` 归一化)， ``"ortho"`` (按 :math:`1*\sqrt{n}` 归一化)。

    返回：
        Tensor， `rfftn()` 函数的结果，result.shape[axes[i]]为s[i].
        当 `a` 是 int16、int32、int64、float16、float32、complex64 类型时，返回值类型为 float32。
        当 `a` 是 float64 或 complex128 类型时，返回值类型为 float64。