mindspore.dataset.audio.melscale_fbanks
=======================================

.. py:function:: mindspore.dataset.audio.melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, norm=NormType.NONE, mel_type=MelType.HTK)

    创建频率变换矩阵。

    参数：
        - **n_freqs** (int) - 要加强或应用的频率数。
        - **f_min** (float) - 最小频率，单位为Hz。
        - **f_max** (float) - 最大频率，单位为Hz。
        - **n_mels** (int) - 梅尔滤波器组数。
        - **sample_rate** (int) - 音频波形的采样频率。
        - **norm** (:class:`mindspore.dataset.audio.NormType` , 可选) - 标准化方法，可以是NormType.NONE或NormType.SLANEY。默认值：NormType.NONE。
        - **mel_type** (:class:`mindspore.dataset.audio.MelType` , 可选) - 使用的标度，可以是MelType.HTK或MelType.SLANEY。默认值：MelType.HTK。

    返回：
        numpy.ndarray，频率变换矩阵，shape为( `n_freqs` , `n_mels` )。

    异常：
        - **TypeError** - 如果 `n_freqs` 的类型不为int。
        - **ValueError** - 如果 `n_freqs` 为负数。
        - **TypeError** - 如果 `f_min` 的类型不为float。
        - **ValueError** - 如果 `f_min` 大于 `f_max` 。
        - **TypeError** - 如果 `f_max` 的类型不为float。
        - **ValueError** - 如果 `f_max` 为负数。
        - **TypeError** - 如果 `n_mels` 的类型不为int。
        - **ValueError** - 如果 `n_mels` 不为正数。
        - **TypeError** - 如果 `sample_rate` 的类型不为int。
        - **ValueError** - 如果 `sample_rate` 不为正数。
        - **TypeError** - 如果 `norm` 的类型不为 :class:`mindspore.dataset.audio.NormType` 。
        - **TypeError** - 如果 `mel_type` 的类型不为 :class:`mindspore.dataset.audio.MelType` 。
