mindspore.ops.ifft
=================================

.. py:function:: mindspore.ops.ifft(input, n=None, dim=-1, norm=None)

    计算输入的一维快速傅里叶逆变换。

    .. note::
        - `ifft` 目前仅用于 `mindscience` 科学计算场景，尚不支持其他使用场景。
        - `ifft` 尚不支持Windows平台。

    参数：
        - **input** (Tensor) - 输入的Tensor。支持数据类型：

          - Ascend/CPU： int16、int32、int64、float16、float32、float64、complex64、complex128。

        - **n** (int, 可选) - 信号长度。如果给定，则在计算 `ifft` 之前 `dim` 轴的大小将被零填充或截断至 `n`。
          默认值： ``None`` , 表示无需对 `input` 进行处理。
        - **dim** (int, 可选) - 进行一维 `ifft` 的维度。默认值： ``-1`` ，对 `input` 的最后一个维度进行变换。
        - **norm** (str, 可选) - 标准化模式。默认值： ``None`` ，采用 ``'backward'`` 。
          三种模式定义为：

          - ``'backward'`` 表示不进行标准化。
          - ``'forward'`` 表示按 :math:`1*n` 标准化。
          - ``'ortho'`` 表示按 :math:`1*\sqrt{n}` 标准化。

    返回： 
        Tensor，一维快速傅里叶逆变换的结果。默认与 `input` 同shape，如果给定 `n` ，则 `dim` 轴的大小改为 `n` 。
        当输入为 int16、int32、int64、float16、float32、complex64 时，返回值类型为complex64。
        当输入为 float64、complex128 时，返回值类型为complex128。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `input` 数据类型不是其中之一: int32、int64、float32、float64、complex64、complex128。
        - **TypeError** - 如果 `dim` 和 `n` 不是int类型。
        - **ValueError** - 如果 `dim` 的值超出： :math:`[-input.ndim, input.ndim)` 范围。
        - **ValueError** - 如果 `n` 的值小于1。
        - **ValueError** - 如果 `norm` 的值不是 ``'backward'`` 、 ``'forward'`` 或 ``'ortho'`` 。
