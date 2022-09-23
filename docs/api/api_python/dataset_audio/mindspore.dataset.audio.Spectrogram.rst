mindspore.dataset.audio.Spectrogram
===================================

.. py:class:: mindspore.dataset.audio.Spectrogram(n_fft=400, win_length=None, hop_length=None, pad=0, window=WindowType.HANN, power=2.0, normalized=False, center=True, pad_mode=BorderType.REFLECT, onesided=True)

    从音频信号创建光谱图。

    参数：
        - **n_fft** (int, 可选) - FFT的大小，创建 `n_fft // 2 + 1` 组滤波器，默认值：400。
        - **win_length** (int, 可选) - 窗口大小，默认值：None，将设置为 `n_fft` 的值。
        - **hop_length** (int, 可选) - STFT窗口之间的跳数长度，默认值：None，将设置为 `win_length//2` 。
        - **pad** (int, 可选) - 信号的双面填充，默认值：0。
        - **window** (WindowType, 可选) - GriffinLim的窗口类型，可以是WindowType.BARTLETT，
          WindowType.BLACKMAN，WindowType.HAMMING，WindowType.HANN或WindowType.KAISER。
          默认值：WindowType.HANN，目前macOS上不支持kaiser窗口。
        - **power** (float, 可选) - 幅度谱图的指数，默认值：2.0。
        - **normalized** (bool, 可选) - 是否在stft之后按幅度归一化。默认值：False。
        - **center** (bool, 可选) - 是否在两侧填充波形，默认值：True。
        - **pad_mode** (BorderType, 可选) - 控制中心为True时使用的填充方法，可以是BorderType.REFLECT、BorderType.CONSTANT、
          BorderType.EDGE、BorderType.SYMMETRIC，默认值：BorderType.REFLECT。
        - **onesided** (bool, 可选) - 控制是否返回一半结果以避免冗余，默认值：True。
    