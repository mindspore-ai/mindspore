mindspore.dataset.audio.melscale_fbanks
=======================================

.. py:function:: mindspore.dataset.audio.melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, norm=NormType.NONE, mel_type=MelType.HTK)

    创建shape为(`n_freqs`, `n_mels`)的频率变换矩阵。

    参数：
        - **n_freqs** (int) - 频率数。
        - **f_min** (float) - 频率的最小值，单位为Hz。
        - **f_max** (float) - 频率的最大值，单位为Hz。
        - **n_mels** (int) - 梅尔滤波器的数量。
        - **sample_rate** (int) - 采样频率（单位：Hz）。
        - **norm** (NormType, 可选) - 规范化的类型，可以是NormType.NONE或NormType.SLANEY。默认值：NormType.NONE
        - **mel_type** (MelType, 可选) - 梅尔滤波器的类型，可以是MelType.HTK或MelType.SLAN。默认值：NormType.SLAN。

    返回：
        numpy.ndarray，频率变换矩阵。
