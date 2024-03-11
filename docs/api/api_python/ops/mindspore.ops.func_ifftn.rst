mindspore.ops.ifftn
=================================

.. py:function:: mindspore.ops.ifftn(input, s=None, dim=None, norm=None)

    计算输入的 `N` 维快速傅里叶逆变换。

    .. note::
        - `ifftn` 目前仅用于 `mindscience` 科学计算场景，尚不支持其他使用场景。
        - `ifftn` 尚不支持Windows平台。

    参数：
        - **input** (Tensor) - 输入的Tensor。支持数据类型：

          - Ascend/CPU： int16、int32、int64、float16、float32、float64、complex64、complex128。

        - **s** (tuple[int], 可选) - 信号长度。如果给定，则在计算 `ifftn` 之前 `dim[i]` 轴的大小将被零填充或截断至 `s[i]`。
          默认值： ``None`` ， 表示无需对 `input` 进行处理。
        - **dim** (tuple[int], 可选) - 指定进行 `ifftn` 的维度。
          默认值：``None`` ，如果 `s` 给定则对最后 `len(s)` 维度进行变换，否则对所有维度进行变换。
        - **norm** (str, 可选) - 标准化模式。默认值： ``None`` ，采用 ``'backward'`` 。
          三种模式定义为：

          - ``'backward'`` 表示不进行标准化。
          - ``'forward'`` 表示按 :math:`1*n` 标准化。
          - ``'ortho'`` 表示按 :math:`1*\sqrt{n}` 标准化。

    返回： 
        Tensor，N维快速傅里叶逆变换的结果。默认与 `input` 同shape，如果给定 `s` ，则 `dim[i]` 轴的大小改为 `s[i]` 。
        当输入为 int16、int32、int64、float16、float32、complex64 时，返回值类型为complex64。
        当输入为 float64、complex128 时，返回值类型为complex128。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `input` 数据类型不是其中之一: int32、int64、float32、float64、complex64、complex128。
        - **TypeError** - 如果 `dim` 和 `s` 不是tuple[int]类型。
        - **ValueError** - 如果 `dim` 中存在超出： :math:`[-input.ndim, input.ndim)` 范围的值。
        - **ValueError** - 如果 `dim` 中存在重复值。
        - **ValueError** - 如果 `s` 中存在小于1的值。
        - **ValueError** - 如果同时给定 `dim` 和 `s`, 但两者shape并不相同。
        - **ValueError** - 如果 `norm` 的值不是 ``'backward'`` 、 ``'forward'`` 或 ``'ortho'`` 。
