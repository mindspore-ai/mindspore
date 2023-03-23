mindspore.dataset.audio.SpectralCentroid
========================================

.. py:class:: mindspore.dataset.audio.SpectralCentroid(sample_rate, n_fft=400, win_length=None, hop_length=None, pad=0, window=WindowType.HANN)

    计算每个通道沿时间轴的频谱中心。

    参数：
        - **sample_rate** (int) - 音频信号的采样率，例如44100 (Hz)。
        - **n_fft** (int, 可选) - FFT的大小，将创建 `n_fft // 2 + 1` 个频段。默认值：400。
        - **win_length** (int, 可选) - 窗口大小。默认值：None，将使用 `n_fft` 。
        - **hop_length** (int, 可选) - STFT窗口之间的跳跃长度。默认值：None，将使用 `win_length // 2` 。
        - **pad** (int, 可选) - 信号两端的填充长度。默认值：0。
        - **window** (:class:`mindspore.dataset.audio.WindowType` , 可选) - 作用于每一帧的窗口函数，可为WindowType.BARTLETT、WindowType.BLACKMAN、
          WindowType.HAMMING、WindowType.HANN或WindowType.KAISER。默认值：WindowType.HANN。

    异常：
        - **TypeError** - 当 `sample_rate` 的类型不为int。
        - **ValueError** - 当 `sample_rate` 为负数。
        - **TypeError** - 当 `n_fft` 的类型不为int。
        - **ValueError** - 当 `n_fft` 不为正数。
        - **TypeError** - 当 `win_length` 的类型不为int。
        - **ValueError** - 当 `win_length` 不为正数。
        - **ValueError** - 当 `win_length` 大于 `n_fft` 。
        - **TypeError** - 当 `hop_length` 的类型不为int。
        - **ValueError** - 当 `hop_length` 不为正数。
        - **TypeError** - 当 `pad` 的类型不为int。
        - **ValueError** - 当 `pad` 为负数。
        - **TypeError** - 当 `window` 的类型不为 :class:`mindspore.dataset.audio.WindowType` 。
        - **RuntimeError** - 当输入音频的shape不为<..., time>。
