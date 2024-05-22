mindspore.scipy.fft.dctn
==========================

.. py:function:: mindspore.scipy.fft.dctn(x, type=2, s=None, axes=None, norm=None)

    计算Tensor `x` 给定维度（轴） `axes` 的N维离散傅里叶变换。

    .. note::
        - `dctn` 目前仅用于 `mindscience` 科学计算场景，尚不支持其他使用场景。
        - `dctn` 尚不支持Windows平台。

    参数：
        - **x** (Tensor) - 要计算离散傅里叶变换的Tensor。支持数据类型：

          - Ascend/CPU：int16、int32、int64、float16、float32、float64、complex64、complex128。

        - **type** (int, 可选) - 离散傅里叶变换变换的种类。当前仅支持类型2离散变换，详见 `‘A Fast Cosine Transform in One and Two Dimensions’,
          by J. Makhoul, IEEE Transactions on acoustics, speech and signal processing vol. 28(1), pp. 27-34, <https://doi.org/10.1109/TASSP.1980.1163351>`_。默认值： ``2`` 。
        - **s** (tuple[int], 可选) - 输出在 `axes` 轴的长度。如果给定，则在计算 `dctn` 之前 `axes[i]` 轴的大小将被零填充或截断至 `s[i]`。
          默认值： ``None`` , 表示无需对 `x` 进行处理。
          请注意，Ascend后端需要传入此参数。
        - **axes** (tuple[int], 可选) - 指定进行 `dctn` 的维度。
          默认值： ``None`` , 如果给定 `s` 则对最后 `len(s)` 维度进行变换，否则对所有维度进行变换。
          请注意，Ascend后端需要传入此参数。
        - **norm** (str, 可选) - 标准化类型，当前仅支持"ortho"。默认值： ``None`` 默认为 ``'ortho'`` 。

    返回：
        Tensor，N维离散傅里叶变换的结果。默认与 `x` 同shape，如果给定 `s`，则 `axes[i]` 轴的大小改为 `s[i]` 。
        当输入为 int16、int32、int64、float16、float32 时，返回值类型为float32。
        当输入为 float64 时，返回值类型为float64。
        当输入为 complex64、complex128 时，返回值类型为 complex64、complex128。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
        - **TypeError** - 如果 `x` 数据类型不属于支持的数据类型。
        - **TypeError** - 如果 `axes` 和 `s` 不是tuple[int]类型。
        - **ValueError** - 如果 `type` 的值不是 `2` 。
        - **ValueError** - 如果 `axes` 中存在超出： :math:`[-x.ndim, x.ndim)` 范围的值。
        - **ValueError** - 如果 `axes` 中存在重复值。
        - **ValueError** - 如果 `s` 中存在小于1的值。
        - **ValueError** - 如果同时给定 `axes` 和 `s`, 但两者shape并不相同。
        - **ValueError** - 如果 `norm` 的值不是 ``'ortho'`` 。
