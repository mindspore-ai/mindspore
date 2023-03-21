mindspore.dataset.audio.MelSpectrogram
======================================

.. py:class:: mindspore.dataset.audio.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=None, hop_length=None, f_min=0.0, f_max=None, pad=0, n_mels=128, window=WindowType.HANN, power=2.0, normalized=False, center=True, pad_mode=BorderType.REFLECT, onesided=True, norm=NormType.NONE, mel_scale=MelType.HTK)

    计算原始音频信号的梅尔频谱。

    参数：
        - **sample_rate** (int, 可选) - 采样频率（单位：Hz），不能小于0。默认值：16000。
        - **n_fft** (int, 可选) -  FFT的大小，创建 `n_fft // 2 + 1` 个频段，应该大于0并且小于输入张量最后一维大小的两倍。默认值：400。
        - **win_length** (int, 可选) - 窗口大小, 应该大于0并且不能大于 `n_fft` 。默认值：None, 会被设置为 `n_ftt` 。
        - **hop_length** (int, 可选) - STFT窗口之间的跳跃长度，应该大于0。默认值：None, 会被设置为 `win_length // 2` 。
        - **f_min** (float, 可选) - 最小频率，不能大于f_max。默认值：0.0。
        - **f_max** (float, 可选) - 最大频率，不能小于0。默认值：None, 会被设置为 `sample_rate // 2` 。
        - **pad** (int, 可选) - 信号两端的填充长度，不能小于0。默认值：0。
        - **n_mels** (int, 可选) - 梅尔滤波器组的数量，不能小于0。默认值：128。
        - **window** (:class:`mindspore.dataset.audio.WindowType` , 可选) - 作用于每一帧的窗口函数。默认值：WindowType.HANN。
        - **power** (float, 可选) - 幅值谱图的指数，应该大于0，例如，1表示能量，2表示功率，等等。默认值：2.0。
        - **normalized** (bool, 可选) - 是否在stft之后按幅度执行标准化。默认值：False。
        - **center** (bool, 可选) - 是否同时在波形两端进行填充。默认值：True。
        - **pad_mode** (BorderType, 可选) - 控制在 `center` 为True时使用的填充方法，可为BorderType.REFLECT、BorderType.CONSTANT、BorderType.EDGE、BorderType.SYMMETRIC。默认值：BorderType.REFLECT。
        - **onesided** (bool, 可选) - 控制是否只返回一半波形，以避免冗余。默认值：True。
        - **norm** (:class:`mindspore.dataset.audio.NormType` , 可选) - 如果为 'slaney' ，则将三角形梅尔权重除以梅尔带的宽度（区域归一化）。默认值：NormType.NONE，不使用标准化。
        - **mel_scale** (:class:`mindspore.dataset.audio.MelType` , 可选) - 要使用的Mel比例，可以是MelType.SLAN或MelType.HTK。默认值：MelType.HTK。

    异常：
        - **TypeError** - 如果 `sample_rate` 的类型不为int。
        - **TypeError** - 如果 `n_fft` 的类型不为int。
        - **TypeError** - 如果 `n_mels` 的类型不为int。
        - **TypeError** - 如果 `f_min` 的类型不为float。
        - **TypeError** - 如果 `f_max` 的类型不为float。
        - **TypeError** - 如果 `window` 的类型不为 :class:`mindspore.dataset.audio.WindowType` 。
        - **TypeError** - 如果 `norm` 的类型不为 :class:`mindspore.dataset.audio.NormType` 。
        - **TypeError** - 如果 `mel_scale` 的类型不为 :class:`mindspore.dataset.audio.MelType` 。
        - **TypeError** - 如果 `power` 的类型不为float。
        - **TypeError** - 如果 `normalized` 的类型不为bool。
        - **TypeError** - 如果 `center` 的类型不为bool。
        - **TypeError** - 如果 `pad_mode` 的类型不为 :class:`mindspore.dataset.audio.BorderType` 。
        - **TypeError** - 如果 `onesided` 的类型不为bool。
        - **TypeError** - 如果 `pad` 的类型不为int。
        - **TypeError** - 如果 `win_length` 的类型不为int。
        - **TypeError** - 如果 `hop_length` 的类型不为int。
        - **ValueError** - 如果 `sample_rate` 为负数。
        - **ValueError** - 如果 `n_ftt` 不为正数。
        - **ValueError** - 如果 `n_mels` 为负数。
        - **ValueError** - 如果 `f_min` 大于 `f_max` 。
        - **ValueError** - 如果 `f_max` 为负数。
        - **ValueError** - 当 `f_max` 为None时, 如果 `f_min` 大于 `sample_rate // 2` 。
        - **ValueError** - 如果 `power` 不为正数。
        - **ValueError** - 如果 `pad` 为负数。
        - **ValueError** - 如果 `win_length` 不为正数。
        - **ValueError** - 如果 `hop_length` 不为正数。
