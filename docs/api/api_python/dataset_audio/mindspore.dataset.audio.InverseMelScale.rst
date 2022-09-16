mindspore.dataset.audio.InverseMelScale
=======================================

.. py:class:: mindspore.dataset.audio.InverseMelScale(n_stft, n_mels=128, sample_rate=16000, f_min=0.0, f_max=None, max_iter=100000, tolerance_loss=1e-05, tolerance_change=1e-08, sgdargs=None, norm=NormType.NONE, mel_type=MelType.HTK)

    使用转换矩阵求解STFT，形成梅尔频率的STFT。

    参数：
        - **n_stft** (int, 可选) - STFT中的滤波器的组数。
        - **n_mels** (int, 可选) - mel滤波器的数量，默认值：128。
        - **sample_rate** (int, 可选) - 音频信号采样频率，默认值：16000。
        - **f_min** (float, 可选) - 最小频率，默认值：0.0。
        - **f_max** (float, 可选) - 最大频率，默认值：None，将设置为 `sample_rate//2` 。
        - **max_iter** (int, 可选) - 最大优化迭代次数，默认值：100000。
        - **tolerance_loss** (float, 可选) - 当达到损失值时停止优化，默认值：1e-5。
        - **tolerance_change** (float, 可选) - 指定损失差异，当达到损失差异时停止优化，默认值：1e-8。
        - **sgdargs** (dict, 可选) - SGD优化器的参数，默认值：None，将设置为{'sgd_lr': 0.1, 'sgd_momentum': 0.9}。
        - **norm** (NormType, 可选) - 标准化方法，可以是NormType.SLANEY或NormType.NONE。默认值：NormType.NONE。
        - **mel_type** (MelType, 可选) - 要使用的Mel比例，可以是MelType.SLAN或MelType.HTK。默认值：MelType.HTK。
    