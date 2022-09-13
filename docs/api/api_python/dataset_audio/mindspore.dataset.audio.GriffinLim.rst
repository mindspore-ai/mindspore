mindspore.dataset.audio.GriffinLim
==================================

.. py:class:: mindspore.dataset.audio.GriffinLim(n_fft=400, n_iter=32, win_length=None, hop_length=None, window_type=WindowType.HANN, power=2, momentum=0.99, length=None, rand_init=True)

    使用GriffinLim算法对音频波形进行近似幅度谱图反演。

    .. math::
        x(n)=\frac{\sum_{m=-\infty}^{\infty} w(m S-n) y_{w}(m S, n)}{\sum_{m=-\infty}^{\infty} w^{2}(m S-n)}

    其中w表示窗口函数，y表示每个帧的重建信号，x表示整个信号。

    参数：
        - **n_fft** (int, 可选) - FFT的长度，默认值：400。
        - **n_iter** (int, 可选) - 相位恢复的迭代次数，默认值：32。
        - **win_length** (int, 可选) - GriffinLim的窗口大小，默认值：None，将设置为 `n_fft` 的值。
        - **hop_length** (int, 可选) - STFT窗口之间的跳数长度，默认值：None，将设置为 `win_length//2` 。
        - **window_type** (WindowType, 可选) - GriffinLim的窗口类型，可以是WindowType.BARTLETT，
          WindowType.BLACKMAN，WindowType.HAMMING，WindowType.HANN或WindowType.KAISER。
          默认值：WindowType.HANN，目前macOS上不支持kaiser窗口。
        - **power** (float, 可选) - 幅度谱图的指数，默认值：2.0。
        - **momentum** (float, 可选) - 快速Griffin-Lim的动量，默认值：0.99。
        - **length** (int, 可选) - 预期输出波形的长度。默认值：None，将设置为stft矩阵的最后一个维度的值。
        - **rand_init** (bool, 可选) - 随机相位初始化或全零相位初始化标志，默认值：True。
    