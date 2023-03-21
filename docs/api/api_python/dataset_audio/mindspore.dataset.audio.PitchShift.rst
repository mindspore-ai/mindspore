mindspore.dataset.audio.PitchShift
==================================

.. py:class:: mindspore.dataset.audio.PitchShift(sample_rate, n_steps, bins_per_octave=12, n_fft=512, win_length=None, hop_length=None, window=WindowType.HANN)

    将波形的音调移动 `n_steps` 步长。

    参数：
        - **sample_rate** (int) - 波形的采样频率（单位：Hz）。
        - **n_steps** (int) - 移动波形的步长。
        - **bins_per_octave** (int, 可选) - 每倍频程的步长。默认值：12。
        - **n_fft** (int, 可选) - FFT的大小，创建 `n_fft // 2 + 1` 个频段。默认值：512。
        - **win_length** (int, 可选) - 窗口大小。默认值：None，将会设置为 `n_fft` 。
        - **hop_length** (int, 可选) - STFT窗口之间的跳跃长度。默认值：None，则将设置为 `win_length // 4` 。
        - **window** (:class:`mindspore.dataset.audio.WindowType` , 可选) - 作用于每一帧的窗口函数。默认值：WindowType.HANN。
      
    异常：
        - **TypeError** - 如果 `sample_rate` 的类型不为int。
        - **TypeError** - 如果 `n_steps` 的类型不为int。
        - **TypeError** - 如果 `bins_per_octave` 的类型不为int。
        - **TypeError** - 如果 `n_fft` 的类型不为int。
        - **TypeError** - 如果 `win_length` 的类型不为int。
        - **TypeError** - 如果 `hop_length` 的类型不为int。
        - **TypeError** - 如果 `window` 的类型不为 :class:`mindspore.dataset.audio.WindowType` 。
        - **ValueError** - 如果 `sample_rate` 为负数。
        - **ValueError** - 如果 `bins_per_octave` 为0。
        - **ValueError** - 如果 `n_fft` 为负数。
        - **ValueError** - 如果 `win_length` 不是正数。
        - **ValueError** - 如果 `hop_length` 不是正数。
