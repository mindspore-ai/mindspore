mindspore.dataset.audio.InverseMelScale
=======================================

.. py:class:: mindspore.dataset.audio.InverseMelScale(n_stft, n_mels=128, sample_rate=16000, f_min=0.0, f_max=None, max_iter=100000, tolerance_loss=1e-5, tolerance_change=1e-8, sgdargs=None, norm=NormType.NONE, mel_type=MelType.HTK)

    使用转换矩阵从梅尔频率STFT求解普通频率的STFT。

    参数：
        - **n_stft** (int) - STFT中的频段数。
        - **n_mels** (int, 可选) - mel滤波器的数量。默认值：128。
        - **sample_rate** (int, 可选) - 音频信号采样频率。默认值：16000。
        - **f_min** (float, 可选) - 最小频率。默认值：0.0。
        - **f_max** (float, 可选) - 最大频率。默认值：None，将设置为 `sample_rate//2` 。
        - **max_iter** (int, 可选) - 最大优化迭代次数。默认值：100000。
        - **tolerance_loss** (float, 可选) - 当达到损失值时停止优化。默认值：1e-5。
        - **tolerance_change** (float, 可选) - 指定损失差异，当达到损失差异时停止优化。默认值：1e-8。
        - **sgdargs** (dict, 可选) - SGD优化器的参数。默认值：None，将设置为{'sgd_lr': 0.1, 'sgd_momentum': 0.9}。
        - **norm** (:class:`mindspore.dataset.audio.NormType` , 可选) - 标准化方法，可以是NormType.SLANEY或NormType.NONE。默认值：NormType.NONE，不使用标准化。
        - **mel_type** (:class:`mindspore.dataset.audio.MelType` , 可选) - 要使用的Mel比例，可以是MelType.SLAN或MelType.HTK。默认值：MelType.HTK。

    异常：
        - **TypeError** - 如果 `n_fft` 的类型不为int。
        - **ValueError** - 如果 `n_ftt` 不为正数。
        - **TypeError** - 如果 `n_mels` 的类型不为int。
        - **ValueError** - 如果 `n_mels` 不为正数。
        - **TypeError** - 如果 `sample_rate` 的类型不为int。
        - **ValueError** - 如果 `sample_rate` 不为正数。
        - **TypeError** - 如果 `f_min` 的类型不为float。
        - **ValueError** - 如果 `f_min` 大于等于 `f_max` 。
        - **TypeError** - 如果 `f_max` 的类型不为float。
        - **ValueError** - 如果 `f_max` 为负数。
        - **TypeError** - 如果 `max_iter` 的类型不为int。
        - **ValueError** - 如果 `max_iter` 为负数。
        - **TypeError** - 如果 `tolerance_loss` 的类型不为float。
        - **ValueError** - 如果 `tolerance_loss` 为负数。
        - **TypeError** - 如果 `tolerance_change` 的类型不为float。
        - **ValueError** - 如果 `tolerance_change` 为负数。
        - **TypeError** - 如果 `sgdargs` 的类型不为dict。
        - **TypeError** - 如果 `norm` 的类型不为 :class:`mindspore.dataset.audio.NormType` 。
        - **TypeError** - 如果 `mel_type` 的类型不为 :class:`mindspore.dataset.audio.MelType` 。
