mindspore.ops.stft
==================

.. py:function:: mindspore.ops.stft(x, n_fft, hop_length=None, win_length=None, window=None, center=True, pad_mode="REFLECT", normalized=False, onesided=None, return_complex=None)

    STFT将信号分割成狭窄的时间间隔，并对每个片段进行傅立叶变换来量化非平稳信号频率和相位随时间的变化。

    忽略批处理维，此操作计算以下表达式：

    .. math::

        X[\omega, m]=\sum_{k=0}^{\text {win_length-1 }}
        \text { window }[k] \text { input }[m \times \text { hop_length }+
        k] \exp \left(-j \frac{2 \pi \cdot \omega k}{\text { win_length }}\right)

    其中 :math:`m` 是滑动窗口的索引，:math:`ω` 是频率，其范围在 :math:`0 \leq \omega < \text{n\_fft}0≤ω<n_fft` 。

    参数：
        - **x** (Tensor) - STFT的时间序列，必须是1-D Tensor或2-D Tensor。
        - **n_fft** (int) - 傅里叶变换的尺寸。
        - **hop_length** (int，可选) - 相邻滑动窗口之间的距离。如果为None，取值视为 :math:`floor(n_fft / 4)` 。默认值：None。
        - **win_length** (int，可选) - 窗口和STFT过滤器的尺寸。如果为None，取值视为 `n_fft` 。默认值：None。
        - **window** (Tensor，可选) - 可选的窗口函数，是一个长度为 `win_length` 的一维Tensor。如果为None，视为所含元素都为1。如果 `win_length` < `n_fft` ，在 `window` 两侧填充1至长度为 `n_fft` 后才生效。默认值：None。
        - **center** (bool，可选) - 是否填充 `x` 两侧。默认值：True。
        - **pad_mode** (str，可选) - `center` 为True的时候指定的填充模式。默认值：“REFLECT”。
        - **normalized** (bool，可选) - 控制是否返回规范化的STFT结果。默认值：False。
        - **onesided** (bool，可选) - 控制是否返回一半的结果，以避免实数输入计算结果的冗余。默认值：None。当 `x` 和 `window` 是实数时取值为True，否则为False。
        - **return_complex** (bool，可选) - 若为True，返回一个复数Tensor。若为False，返回一个实数Tensor，
          且其具有额外的最后一维以表示实部和虚部。默认值：None。当 `x` 或 `window` 为复数时取值为True，否则为False。

    返回：
        - **output** (Tensor) - 包含STFT计算的结果的Tensor。

          - 如果 `return_complex` 为True，则返回一个复数Tensor，shape为 :math:`(*, N, T)` 。
          - 如果 `return_complex` 为False，则返回一个实数Tensor，shape为 :math:`(*, N, T, 2)` 。

          `N` 为傅立叶变换的尺寸，取值受参数 `onesided` 影响：
          - 如果 `onesided` 为False， :math:`N = n_fft` 。
          - 如果 `onesided` 为True， :math:`N = n_fft // 2 + 1` 。
            
          `T` 为使用的总帧数，计算公式：:math:`T = 1 + (len - n_fft) / hop_length` ，其中 :math:`len` 取值受 `center` 影响：
          - 如果 `center` 为False，则 :math:`len = signal_length` 。
          - 如果 `center` 为True，则 :math:`len = signal_length + (n_fft // 2) * 2` 。

          其中signal_length为信号长度，取值 :math:`x.shape[-1]` 。     

    异常：
        - **TypeError** -  `x` 不是1-D或2-D Tensor。
        - **TypeError** -  `window` 不是1-DTensor。
        - **TypeError** -  `center` 、 `normalized` 、 `onesided` 和 `return_complex` 中任意一个被指定了非布尔类型的值。
        - **TypeError** -  `pad_mode` 被指定了非str类型的值。
        - **TypeError** -  `n_fft` 、 `hop_length` 和 `hop_length` 中任意一个不是int类型。
