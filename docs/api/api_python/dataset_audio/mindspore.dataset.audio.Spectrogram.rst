mindspore.dataset.audio.Spectrogram
===================================

.. py:class:: mindspore.dataset.audio.Spectrogram(n_fft=400, win_length=None, hop_length=None, pad=0, window=WindowType.HANN, power=2.0, normalized=False, center=True, pad_mode=BorderType.REFLECT, onesided=True)

    从音频信号创建其频谱。

    参数：
        - **n_fft** (int, 可选) - FFT的大小，将创建 `n_fft // 2 + 1` 个频段。默认值：400。
        - **win_length** (int, 可选) - 窗口大小。默认值：None，将使用 `n_fft` 。
        - **hop_length** (int, 可选) - STFT窗口之间的跳跃长度。默认值：None，将使用 `win_length//2` 。
        - **pad** (int, 可选) - 信号两端的填充长度。默认值：0。
        - **window** (:class:`mindspore.dataset.audio.WindowType` , 可选) - 作用于每一帧的窗口函数，可为WindowType.BARTLETT、WindowType.BLACKMAN、
          WindowType.HAMMING、WindowType.HANN或WindowType.KAISER。当前，在macOS上暂不支持Kaiser窗。默认值：WindowType.HANN。
        - **power** (float, 可选) - 幅度谱图的指数，必须非负，例如1代表能量谱，2代表功率谱等。默认值：2.0。
        - **normalized** (bool, 可选) - 是否在stft之后按幅度执行标准化。默认值：False。
        - **center** (bool, 可选) - 是否同时在波形两端进行填充。默认值：True。
        - **pad_mode** (:class:`mindspore.dataset.audio.BorderType` , 可选) - 控制在 `center` 为True时使用的填充方法，可为BorderType.REFLECT、BorderType.CONSTANT、
          BorderType.EDGE、BorderType.SYMMETRIC。默认值：BorderType.REFLECT。
        - **onesided** (bool, 可选) - 控制是否只返回一半波形，以避免冗余。默认值：True。

    异常：
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
        - **TypeError** - 当 `power` 的类型不为float。
        - **ValueError** - 当 `power` 为负数。
        - **TypeError** - 当 `normalized` 的类型不为bool。
        - **TypeError** - 当 `center` 的类型不为bool。
        - **TypeError** - 当 `pad_mode` 的类型不为 :class:`mindspore.dataset.audio.BorderType` 。
        - **TypeError** - 当 `onesided` 的类型不为bool。
        - **RuntimeError** - 当输入音频的shape不为<..., time>。
