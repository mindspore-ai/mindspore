mindspore.dataset.audio.GriffinLim
==================================

.. py:class:: mindspore.dataset.audio.GriffinLim(n_fft=400, n_iter=32, win_length=None, hop_length=None, window_type=WindowType.HANN, power=2.0, momentum=0.99, length=None, rand_init=True)

    使用Griffin-Lim算法从线性幅度频谱图中计算信号波形。

    有关Griffin-Lim算法更多的描述，详见论文 `A fast Griffin-Lim algorithm <https://doi.org/10.1109/WASPAA.2013.6701851>`_
    与 `Signal estimation from modified short-time Fourier transform <https://doi.org/10.1109/ICASSP.1983.1172092>`_ 。

    参数：
        - **n_fft** (int, 可选) - FFT的长度。默认值：400。
        - **n_iter** (int, 可选) - 相位恢复的迭代次数。默认值：32。
        - **win_length** (int, 可选) - GriffinLim的窗口大小。默认值：None，将设置为 `n_fft` 的值。
        - **hop_length** (int, 可选) - STFT窗口之间的跳数长度。默认值：None，将设置为 `win_length//2` 。
        - **window_type** (:class:`mindspore.dataset.audio.WindowType` , 可选) - GriffinLim的窗口类型，可以是WindowType.BARTLETT，
          WindowType.BLACKMAN，WindowType.HAMMING，WindowType.HANN或WindowType.KAISER。
          默认值：WindowType.HANN，目前macOS上不支持kaiser窗口。
        - **power** (float, 可选) - 幅度谱图的指数。默认值：2.0。
        - **momentum** (float, 可选) - 快速Griffin-Lim的动量。默认值：0.99。
        - **length** (int, 可选) - 预期输出波形的长度。默认值：None，将设置为stft矩阵的最后一个维度的值。
        - **rand_init** (bool, 可选) - 随机相位初始化或全零相位初始化标志。默认值：True。
    
    异常：
        - **TypeError** - 如果 `n_fft` 的类型不为int。
        - **ValueError** - 如果 `n_ftt` 不为正数。
        - **TypeError** - 如果 `n_iter` 的类型不为int。
        - **ValueError** - 如果 `n_mels` 不为正数。
        - **TypeError** - 如果 `win_length` 的类型不为int。
        - **ValueError** - 如果 `win_length` 为负数。
        - **TypeError** - 如果 `hop_length` 的类型不为int。
        - **ValueError** - 如果 `hop_length` 为负数。
        - **TypeError** - 如果 `window_type` 的类型不为 :class:`mindspore.dataset.audio.WindowType` 。
        - **TypeError** - 如果 `power` 的类型不为float。
        - **ValueError** - 如果 `power` 不为正数。
        - **TypeError** - 如果 `momentum` 的类型不为float。
        - **ValueError** - 如果 `momentum` 为负数。
        - **TypeError** - 如果 `length` 的类型不为int。
        - **ValueError** - 如果 `length` 为负数。
        - **TypeError** - 如果 `rand_init` 的类型不为bool。        
        - **RuntimeError** - 当 `n_fft` 指定的FFT长度不小于 `length` 指定的输出波形长度。
        - **RuntimeError** - 当 `win_length` 指定的窗口长度不小于 `n_fft` 指定的FFT长度。
