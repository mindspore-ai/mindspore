mindspore.ops.ihfft
=================================

.. py:function:: mindspore.ops.ihfft(input, n=None, dim=-1, norm=None)

    计算具有厄密对称输入的一维快速傅里叶逆变换。

    .. note::
        - `ihfft` 目前仅用于 `mindscience` 科学计算场景，尚不支持其他使用场景。
        - `ihfft` 尚不支持Windows平台。

    参数：
        - **input** (Tensor) - 输入的Tensor。支持数据类型：

          - Ascend/CPU： int16、int32、int64、float16、float32、float64。

        - **n** (int, 可选) - 输出在 `dim` 轴的长度。如果给定，则在计算 `ihfft` 之前 `dim` 轴的大小将被零填充或截断至 `n`。
          默认值： ``None`` , 表示无需对 `input` 进行处理。
        - **dim** (int, 可选) - 进行一维 `ihfft` 的维度。默认值： ``-1`` ，表示对 `input` 的最后一个维度进行变换。
        - **norm** (str, 可选) - 标准化模式。默认值： ``None`` ，采用 ``'backward'`` 。
          三种模式定义为：

          - ``'backward'`` 表示按 :math:`1/n` 标准化。
          - ``'forward'`` 表示不进行标准化。
          - ``'ortho'`` 表示按 :math:`1/\sqrt{n}` 标准化。

    返回： 
        Tensor， `ihfft` 的结果。如果给定 `n` ，则 `dim` 轴的大小改为 :math:`n // 2 + 1`, 否则 :math:`input.shape[dim] // 2 + 1`。
        当输入为 int16、int32、int64、float16、float32 时，返回值类型为 complex64。
        当输入为 float64 时，返回值类型为 complex128

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `input` 数据类型不是其中之一: int32、int64、float32、float64。
        - **TypeError** - 如果 `dim` 和 `n` 不是int类型。
        - **ValueError** - 如果 `dim` 的值超出： :math:`[-input.ndim, input.ndim)` 范围。
        - **ValueError** - 如果 `n` 的值小于1。
        - **ValueError** - 如果 `norm` 的值不是 ``'backward'`` 、 ``'forward'`` 或 ``'ortho'`` 。
