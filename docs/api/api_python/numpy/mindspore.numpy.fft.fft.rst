mindspore.numpy.fft.fft
=======================

.. py:function:: mindspore.numpy.fft.fft(a, n=None, axis=-1, norm=None)

    计算 `a` 一维离散傅里叶变换。

    请参阅 :func:`mindspore.ops.fft` 以获取更多详细信息。
    区别在于 `a` 对应于 `input` ，`axis` 对应于 `dim` 。

    参数：
        - **a** (Tesor) - 输入Tensor。 支持的dtype： - Ascend/CPU:int16, int32, int64, float16, float32, float64, complex64, complex128。
        - **n** (int, 可选) - 结果 `axis` 轴变换为的长度。如果给定，在计算 `fft` 之前， `axis` 轴的大小将进行零填充或截断为 `n` 。默认值： ``None`` ，表示不需要处理 `a` 。
        - **axis** (int, 可选) - 计算 `fft` 所沿的轴。 默认值： ``-1`` ，表示沿 `a` 的最后一个轴。
        - **norm** (str, 可选) - 归一化模式。 默认值： ``None`` ，表示 ``"backward"`` 。 三种模式的定义如下， ``"backward"`` (无归一化)。 - ``"forward"`` (按 :math:`1/n` 归一化)。 - ``"ortho"`` (按 :math:`1/\sqrt{n}` 归一化)。

    返回：
        Tensor， `fft()` 函数的结果。默认与 `a` shape相同。
        如果给定 `n` ，则 `axis` 的大小将更改为 `n` 。
        当 `a` 是 int16、int32、int64、float16、float32、complex64 类型时，返回值类型为 complex64。
        当 `a` 是 float64 或 complex128 类型时，返回值类型为 complex128。