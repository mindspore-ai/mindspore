mindspore.ops.FFTWithSize
=========================

.. py:class:: mindspore.ops.FFTWithSize(signal_ndim, inverse, real, norm="backward", onesided=True, signal_sizes=())

    傅里叶变换，可以对参数进行调整，以实现FFT/IFFT/RFFT/IRFFT。

    对于FFT，它计算以下表达式：

    .. math::
        X[\omega_1, \dots, \omega_d] =
            \sum_{n_1=0}^{N_1-1} \dots \sum_{n_d=0}^{N_d-1} x[n_1, \dots, n_d]
             e^{-j\ 2 \pi \sum_{i=0}^d \frac{\omega_i n_i}{N_i}},

    其中， :math:`d` = `signal_ndim` 是信号的维度，:math:`N_i` 则是信号第 :math:`i` 个维度的大小。
    
    对于IFFT，它计算以下表达式：

    .. math::
        X[\omega_1, \dots, \omega_d] =
            \frac{1}{\prod_{i=1}^d N_i} \sum_{n_1=0}^{N_1-1} \dots \sum_{n_d=0}^{N_d-1} x[n_1, \dots, n_d]
             e^{\ j\ 2 \pi \sum_{i=0}^d \frac{\omega_i n_i}{N_i}},

    其中， :math:`d` = `signal_ndim` 是信号的维度，:math:`N_i` 则是信号第 :math:`i` 维的大小。

    .. note::
        - FFT/IFFT要求complex64或complex128类型的输入，返回complex64或complex128类型的输出。
        - RFFT要求bool, uint8, int8, int16, int32, int64, float32或float64类型的输入，
          返回complex64或complex128类型的输出。
        - IRFFT要求complex64或complex128类型的输入，返回float32或float64类型的输出。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **signal_ndim** (int) - 表示每个信号中的维数，控制着傅里叶变换的维数，其值只能为1、2或3。
        - **inverse** (bool) - 表示该操作是否为逆变换，用以选择FFT 和 RFFT 或 IFFT 和 IRFFT。

          - 如果为 ``True`` ，则为IFFT 和 IRFFT。
          - 如果为 ``False`` ，FFT 和 RFFT。

        - **real** (bool) - 表示该操作是否为实变换，与 `inverse` 共同决定具体的变换模式：

          - `inverse` 为 ``False`` ， `real` 为 ``False`` ：对应FFT模式。
          - `inverse` 为 ``True`` ， `real` 为 ``False`` ：对应IFFT模式。
          - `inverse` 为 ``False`` ， `real` 为 ``True`` ：对应RFFT模式。
          - `inverse` 为 ``True`` ，  `real` 为 ``True``  ：对应IRFFT模式。
  
        - **norm** (str，可选) - 表示该操作的规范化方式，可选值：[ ``"backward"`` , ``"forward"`` , ``"ortho"`` ]。默认值： ``"backward"`` 。
  
          - "backward"，正向变换不缩放，逆变换按 :math:`1/n` 缩放，其中 `n` 表示输入 `x` 的元素数量。。
          - "ortho"，正向变换与逆变换均按 :math:`1/\sqrt n` 缩放。
          - "forward"，正向变换按 :math:`1/n` 缩放，逆变换不缩放。
  
        - **onesided** (bool，可选) - 控制输入是否减半以避免冗余。默认值： ``True`` 。
        - **signal_sizes** (tuple，可选) - 原始信号的大小（RFFT变换之前的信号，不包含batch这一维），只有在IRFFT模式下和设置 `onesided` 为True时需要该参数，需要满足
          以下条件。默认值： ``()`` 。

          - `signal_sizes` 的长度等于IRFFT的 `signal_ndim` ： :math:`len(signal\_sizes)=signal\_ndim` 。
          - `signal_sizes` 的最后一个维度除以2等于IRFFT输入的最后一个维度： :math:`signal\_size[-1]/2+1=x.shape[-1]` 。
          - 除了最后一个维度外， `signal_sizes` 的维度与输入shape完全相同： :math:`signal\_sizes[:-1]=x.shape[:-1]` 。
 
    输入：
        - **x** (Tensor) - 输入Tensor的维数必须大于或等于 `signal_ndim` 。

    输出：
        Tensor，表示复数到复数、实数到复数或复数到实数傅里叶变换的结果。

    异常：
        - **TypeError** - 如果FFT/IFFT/IRFF的输入类型不是以下类型之一：complex64、complex128。
        - **TypeError** - 如果输入的类型不是Tensor。
        - **ValueError** - 如果输入 `x` 的维度小于 `signal_ndim` 。
        - **ValueError** - 如果 `signal_ndim` 大于3或小于1。
        - **ValueError** - 如果 `norm` 取值不是"backward"、"forward"或"ortho"。