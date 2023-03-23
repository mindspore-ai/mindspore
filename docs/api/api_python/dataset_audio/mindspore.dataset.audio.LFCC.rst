mindspore.dataset.audio.LFCC
============================

.. py:class:: mindspore.dataset.audio.LFCC(sample_rate=16000, n_filter=128, n_lfcc=40, f_min=0.0, f_max=None, dct_type=2, norm=NormMode.ORTHO, log_lf=False, speckwargs=None)

    计算音频信号的线性频率倒谱系数。
    
    .. note:: 待处理音频shape需为<..., time>。

    参数：
        - **sample_rate** (int, 可选) - 音频信号的采样率。默认值：16000。
        - **n_filter** (int, 可选) - 要应用的线性滤波器数量。默认值：128。
        - **n_lfcc** (int, 可选) - 要保留的线性频率倒谱系数数。默认值：40。
        - **f_min** (float, 可选) - 最小频率。默认值：0.0。
        - **f_max** (float, 可选) - 最大频率。默认值：None，会被设置为 `sample_rate // 2` 。
        - **dct_type** (int, 可选) - 要使用的离散余弦变换的类型。该值只能为2。默认值：2。
        - **norm** (:class:`mindspore.dataset.audio.NormMode` , 可选) - 要使用的标准化方法。默认值：NormMode.ORTHO。
        - **log_lf** (bool, 可选) - 是否使用对数-线性频谱图而不是以分贝为刻度的频谱图。默认值：False。
        - **speckwargs** (dict, 可选) - :class:`mindspore.dataset.audio.Spectrogram` 接口的参数。默认值：None，会被设置为包含以下字段的字典

          - 'n_fft': 400
          - 'win_length': n_fft
          - 'hop_length': win_length // 2
          - 'pad': 0
          - 'window': WindowType.HANN
          - 'power': 2.0
          - 'normalized': False
          - 'center': True
          - 'pad_mode': BorderType.REFLECT
          - 'onesided': True

    异常：
        - **TypeError** - 如果 `sample_rate` 的类型不为int。
        - **TypeError** - 如果 `n_filter` 的类型不为int。
        - **TypeError** - 如果 `n_lfcc` 的类型不为int。
        - **TypeError** - 如果 `norm` 的类型不为 :class:`mindspore.dataset.audio.NormMode` 。
        - **TypeError** - 如果 `log_lf` 的类型不为bool。
        - **TypeError** - 如果 `speckwargs` 的类型不为dict。
        - **ValueError** - 如果 `sample_rate` 为0。
        - **ValueError** - 如果 `n_lfcc` 小于0。
        - **ValueError** - 如果 `f_min` 大于 `f_max` 。
        - **ValueError** - 当 `f_max` 为None时，如果 `f_min` 大于 `sample_rate // 2` 。
        - **ValueError** - 如果 `dct_type` 不为2。
