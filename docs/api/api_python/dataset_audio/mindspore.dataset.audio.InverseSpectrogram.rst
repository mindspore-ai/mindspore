mindspore.dataset.audio.InverseSpectrogram
==========================================

.. py:class:: mindspore.dataset.audio.InverseSpectrogram(length=None, n_fft=400, win_length=None, hop_length=None, pad=0, window=WindowType.HANN, normalized=False, center=True, pad_mode=BorderType.REFLECT, onesided=True)

    计算输入频谱的反向频谱，以恢复原始音频信号。

    参数：
        - **length** (int, 可选) - 波形的输出长度，必须是非负数。默认值：None，表示输出整个波形。
        - **n_fft** (int, 可选) - FFT的大小，创建 `n_FFT//2+1` 个频段，应该大于0。默认值：400。
        - **win_length** (int, 可选) - 窗口大小，应该大于0。默认值：None，将被设为 `n_fft` 。
        - **hop_length** (int, 可选) - STFT窗口之间的跳跃长度，应该大于0。默认值：None，将被设为 `win_length // 2` 。
        - **pad** (int, 可选) - 信号两端的填充长度，不能小于0。默认值：0。
        - **window** (:class:`mindspore.dataset.audio.WindowType` , 可选) - 作用于每一帧的窗口函数。默认值：WindowType.HANN。
        - **normalized** (bool, 可选) - 是否在stft之后按幅度执行标准化。默认值：False。
        - **center** (bool, 可选) - 是否同时在波形两端进行填充。默认值：True。
        - **pad_mode** (:class:`mindspore.dataset.audio.BorderType` , 可选) - 控制在 `center` 为True时使用的填充方法，可为BorderType.REFLECT、BorderType.CONSTANT、
          BorderType.EDGE、BorderType.SYMMETRIC。默认值：BorderType.REFLECT。
        - **onesided** (bool, 可选) - 控制是否只返回一半波形，以避免冗余。默认值：True。

    异常：
        - **TypeError** - 如果 `length` 的类型不为int。
        - **ValueError** - 如果 `length` 为负数。
        - **TypeError** - 如果 `n_fft` 的类型不为int。
        - **ValueError** - 如果 `n_fft` 不为正数。
        - **TypeError** - 如果 `win_length` 的类型不为int。
        - **ValueError** - 如果 `win_length` 不为正数。
        - **TypeError** - 如果 `hop_length` 的类型不为int。
        - **ValueError** - 如果 `hop_length` 不为正数。
        - **TypeError** - 如果 `pad` 的类型不为int。
        - **ValueError** - 如果 `pad` 为负数。
        - **TypeError** - 如果 `window` 的类型不为 :class:`mindspore.dataset.audio.WindowType` 。
        - **TypeError** - 如果 `normalized` 的类型不为bool。
        - **TypeError** - 如果 `center` 的类型不为bool。
        - **TypeError** - 如果 `pad_mode` 的类型不为 :class:`mindspore.dataset.audio.BorderType` 。
        - **TypeError** - 如果 `onesided` 的类型不为bool。
