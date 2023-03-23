mindspore.dataset.audio.linear_fbanks
=====================================

.. py:function:: mindspore.dataset.audio.linear_fbanks(n_freqs, f_min, f_max, n_filter, sample_rate)

    创建一个线性三角滤波器组。

    参数：
        - **n_freqs** (int) - 欲加强/作用的频率数。
        - **f_min** (float) - 最小频率，单位赫兹。
        - **f_max** (float) - 最大频率，单位赫兹
        - **n_filter** (int) - 线性三角滤波器数目。
        - **sample_rate** (int) - 音频波形的采样率。

    返回：
        numpy.ndarray，线性三角滤波器组。

    异常：
        - **TypeError** - 如果 `n_freqs` 的类型不为int。
        - **ValueError** - 如果 `n_freqs` 为负数。
        - **TypeError** - 如果 `f_min` 的类型不为float。
        - **ValueError** - 如果 `f_min` 为负数。
        - **TypeError** - 如果 `f_max` 的类型不为float。
        - **ValueError** - 如果 `f_max` 为负数。
        - **ValueError** - 如果 `f_min` 大于 `f_max`。
        - **TypeError** - 如果 `n_filter` 的类型不为int。
        - **ValueError** - 如果 `n_filter` 不为正数。
        - **TypeError** - 如果 `sample_rate` 的类型不为int。
        - **ValueError** - 如果 `sample_rate` 不为正数。
