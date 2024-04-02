mindspore.scipy.fft.dct
==========================

.. py:function:: mindspore.scipy.fft.dct(x, type=2, n=None, axis=-1, norm=None)

    计算Tensor `x` 给定维度（轴） `axis` 的离散傅里叶变换。

    .. note::
        - 当前仅支持type 2类型离散傅里叶变换。
        - `dct` 目前仅用于 `mindscience` 科学计算场景，尚不支持其他使用场景。
        - `dct` 尚不支持Windows平台。

    参数：
        - **x** (Tensor) - 要计算离散傅里叶变换的Tensor。
        - **type** (int, 可选) - 离散傅里叶变换变换的种类。可选值：{1, 2, 3, 4}，详见 `‘A Fast Cosine Transform in One and Two Dimensions’,
          by J. Makhoul, IEEE Transactions on acoustics, speech and signal processing vol. 28(1), pp. 27-34, <https://doi.org/10.1109/TASSP.1980.1163351>`_。默认值： ``2`` 。
        - **n** (int, 可选) - 给定维度（轴）参与离散傅里叶变换的元素数量。必须是非负。
          如果 :math:`n < x.shape[axis]`，则截断后计算，如果 :math:`n > x.shape[axis]` ，则补0后计算。
          默认值： ``None``，默认为 `x.shape[axis]` 。
        - **axis** (int, 可选) - 进行离散傅里叶变换的维度（轴）。
          默认值： ``-1``。
        - **norm** (str, 可选) - 标准化类型，可选值：{"BACKWARD", "FORWARD", "ORTHO"}。默认值： ``None`` 默认为 ``'BACKWARD'`` 。

    返回：
        Tensor， `x` 进行离散傅里叶变换的结果，变换结果的数据类型为float32/64， `axis` 轴大小为 `n`，其余轴大小与 `x` 相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
        - **TypeError** - 如果 `axis` 或 `n` 不是int类型。
        - **ValueError** - 如果 `axis` 的值超出： :math:`[-x.ndim, x.ndim)` 范围。
        - **ValueError** - 如果 `n` 的值小于1。
        - **ValueError** - 如果 `norm` 的值不是 ``BACKWARD``、 ``FORWARD`` 或 ``ORTHO``。
