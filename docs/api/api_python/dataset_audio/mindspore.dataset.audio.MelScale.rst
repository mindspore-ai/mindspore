mindspore.dataset.audio.MelScale
================================

.. py:class:: mindspore.dataset.audio.MelScale(n_mels=128, sample_rate=16000, f_min=0.0, f_max=None, n_stft=201, norm=NormType.NONE, mel_type=MelType.HTK)

    将普通STFT转换为梅尔尺度的STFT。

    参数：
        - **n_mels** (int, 可选) - 梅尔滤波器的数量。默认值：128。
        - **sample_rate** (int, 可选) - 音频信号采样速率。默认值：16000（单位：Hz）。
        - **f_min** (float, 可选) - 最小频率。默认值：0.0。
        - **f_max** (float, 可选) - 最大频率。默认值：None，将设置为 `sample_rate//2` 。
        - **n_stft** (int, 可选) - STFT中的频段数。默认值：201。
        - **norm** (NormType, 可选) - 标准化方法，可以是NormType.SLANEY或NormType.NONE。默认值：NormType.NONE。
          若采用NormType.SLANEY，则三角梅尔权重将被除以梅尔频带的宽度。
        - **mel_type** (MelType, 可选) - 要使用的Mel比例，可以是MelType.SLAN或MelType.HTK。默认值：MelType.HTK。
