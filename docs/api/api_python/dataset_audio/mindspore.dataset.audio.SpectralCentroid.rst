mindspore.dataset.audio.SpectralCentroid
========================================

.. py:class:: mindspore.dataset.audio.SpectralCentroid(sample_rate, n_fft=400, win_length=None, hop_length=None, pad=0, window=WindowType.HANN)

    从音频信号创建光谱质心。

    参数：
        - **sample_rate** (int) - 波形的采样率，例如44100 (Hz)。
        - **n_fft** (int, 可选) - FFT的大小，创建n_fft // 2 + 1 bins。默认值：400。
        - **win_length** (int, 可选) - 窗口大小，默认值：None，将设置为 `n_fft` 的值。
        - **hop_length** (int, 可选) - STFT窗口之间的跳数长度，默认值：None，将设置为 `win_length//2` 。
        - **pad** (int, 可选) - 信号的两侧填充数量，默认值：0。
        - **window** (WindowType, 可选) - 窗口函数，可以是WindowType.BARTLETT、WindowType.BLACKMAN、
          WindowType.HAMMING、WindowType.HANN或WindowType.KAISER。默认值：WindowType.HANN。
