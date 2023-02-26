mindspore.ops.STFT
==================

.. py:class:: mindspore.ops.STFT(n_fft, hop_length, win_length, normalized, onesided, return_complex)

    应用短时傅里叶变换（STFT）于输入信号。

    STFT将信号分割成狭窄的时间间隔，并对每个片段进行傅立叶变换来量化非平稳信号频率和相位随时间的变化。

    更多参考详见 :func:`mindspore.ops.stft` 。
