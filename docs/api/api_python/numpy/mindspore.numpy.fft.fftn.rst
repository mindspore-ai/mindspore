mindspore.numpy.fft.fftn
========================

.. py:function:: mindspore.numpy.fft.fftn(a, s=None, axes=None, norm=None)

    计算 `a` 的N维离散傅里叶变换。

    请参阅 :func:`mindspore.ops.fftn` 以获取更多详细信息。 区别在于 `a` 对应于 `input` ， `axes` 对应于 `dim` 。

    参数：
        - **a** (Tensor) - 输入Tensor。 支持的dtype： - Ascend/CPU:int16, int32, int64, float16, float32, float64, complex64, complex128。
        - **s** (tuple[int], 可选) - 结果中 `axes` 轴变换为的长度。 如果给定，输入将在计算 `fftn` 之前进行零填充或截断为长度 `s` 。 默认值： ``None`` ，表示不处理 `a` 。
        - **axes** (tuple[int], 可选) - 计算 `fftn` 所沿的轴。 默认值： ``None`` ，表示变换 `a` 的所有维度，或者如果给定 `s` ，则为最后的 `len(s)` 维。
        - **norm** (str, 可选) - 归一化模式。 默认值： ``None`` ，表示 ``"backward"`` 。 三种模式的定义如下， ``"backward"`` (无归一化)， ``"forward"`` (按 :math:`1/n` 归一化)， ``"ortho"`` (按 :math:`1/\sqrt{n}` 归一化)。

    返回：
        Tensor， `fftn()` 函数的结果。默认与 `a` shape相同。
        如果给定 `s` ，则 `axes[i]` 轴的大小将更改为 `s[i]` 。
        当 `a` 是 int16、int32、int64、float16、float32、complex64 类型时，返回值类型为 complex64。
        当 `a` 是 float64 或 complex128 类型时，返回值类型为 complex128。