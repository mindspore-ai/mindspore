mindspore.scipy.fft.idct
==========================

.. py:function:: mindspore.scipy.fft.idct(x, type=2, n=None, axis=-1, norm=None)

    计算Tensor `x` 给定维度（轴） `axis` 的逆离散傅里叶变换。

    .. note::
        - 当前仅支持type 2类型逆离散傅里叶变换。
        - 标准化类型 `norm` 当前仅支持 ``'ORTHO'`` 正交标准化。
        - `idct` 目前仅用于 `mindscience` 科学计算场景，尚不支持其他使用场景。
        - `idct` 尚不支持Windows平台。

    参数：
        - **x** (Tensor) - 要计算逆离散傅里叶变换的Tensor。
        - **type** (int, 可选) - 逆离散傅里叶变换变换的种类。可选值：{1, 2, 3, 4}，详见 `‘A Fast Cosine Transform in One and Two Dimensions’,
          by J. Makhoul, IEEE Transactions on acoustics, speech and signal processing vol. 28(1), pp. 27-34, <https://doi.org/10.1109/TASSP.1980.1163351>`_。默认值： ``2`` 。
        - **n** (int, 可选) - 给定维度（轴）参与离散傅里叶变换的元素数量。必须是非负。
          如果 :math:`n < x.shape[axis]`，则截断后计算，如果 :math:`n > x.shape[axis]` ，则补0后计算。
          默认值： ``None``，默认为 `x.shape[axis]` 。
        - **axis** (int, 可选) - 进行逆离散傅里叶变换的维度（轴）。
          默认值： ``-1``。
        - **norm** (str, 可选) - 标准化类型。默认值： ``None`` ，默认为 ``'ORTHO'`` （正交标准化），目前只支持 ``'ORTHO'`` 。

    返回：
        Tensor， `x` 进行逆离散傅里叶变换的结果。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
        - **TypeError** - 如果 `axis` 或 `n` 不是int类型。
        - **ValueError** - 如果 `axis` 的值超出： :math:`[-x.ndim, x.ndim)` 范围。
        - **ValueError** - 如果 `n` 的值小于1。
        - **ValueError** - 如果 `norm` 的值不是 ``'ORTHO'``。
