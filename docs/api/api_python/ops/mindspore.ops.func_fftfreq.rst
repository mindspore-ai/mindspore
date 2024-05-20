mindspore.ops.fftfreq
=================================

.. py:function:: mindspore.ops.fftfreq(n, d=1.0, dtype=None)

    计算大小为 `n` 的信号的离散傅里叶变换采样频率。
    例如，给定长度 `n` 和样本间距 `d` ，返回的结果 `f` 为：
    
    .. math::
        f = [0, 1, ..., (n - 1) // 2, -(n // 2), ..., -1] / (d * n)
        
    .. note::
        - `fftfreq` 目前仅用于 `mindscience` 科学计算场景，尚不支持其他使用场景。
        - `fftfreq` 尚不支持Windows平台。

    参数：
        - **n** (int) - 窗口长度。
        - **d** (float, 可选) - 样本间距（采样率的倒数）。默认值： ``1.0`` 。
        - **dtype** (mindspore.dtype, 可选) - 输出数据类型，目前只支持float16、float32、float64、complex64与complex128。
          默认值： ``None`` ，代表float32。

    返回： 
        Tensor， 长度为 `n` 的一维Tensor，包含样本频率。

    异常：
        - **ValueError** - 如果 `n` 的值小于1。
