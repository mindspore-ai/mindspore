mindspore.dataset.audio.TimeStretch
===================================

.. py:class:: mindspore.dataset.audio.TimeStretch(hop_length=None, n_freq=201, fixed_rate=None)

    以给定的比例拉伸音频短时傅里叶（Short Time Fourier Transform, STFT）频谱的时域，但不改变音频的音高。

    .. note:: 待处理音频shape需为<..., freq, time, complex=2>。第零维代表实部，第一维代表虚部。

    参数：
        - **hop_length** (int, 可选) - STFT窗之间每跳的长度，即连续帧之间的样本数。默认值：None，表示取 `n_freq - 1` 。
        - **n_freq** (int, 可选) - STFT中的滤波器组数。默认值：201。
        - **fixed_rate** (float, 可选) - 频谱在时域加快或减缓的比例。默认值：None，表示保持原始速率。

    异常：
        - **TypeError** - 当 `hop_length` 的类型不为int。
        - **ValueError** - 当 `hop_length` 不为正数。
        - **TypeError** - 当 `n_freq` 的类型不为int。
        - **ValueError** - 当 `n_freq` 不为正数。
        - **TypeError** - 当 `fixed_rate` 的类型不为float。
        - **ValueError** - 当 `fixed_rate` 不为正数。
        - **RuntimeError** - 当输入音频的shape不为<..., freq, num_frame, complex=2>。

    .. image:: time_stretch_rate1.5.png

    .. image:: time_stretch_original.png

    .. image:: time_stretch_rate0.8.png
