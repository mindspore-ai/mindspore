mindspore.ops.STFT
==================

.. py:class:: mindspore.ops.STFT(n_fft, hop_length, win_length, normalized, onesided, return_complex)

    通过STFT量化非平稳信号频率和相位随时间的变化。

    参数：
        - **n_fft** (int) - 傅里叶变换的尺寸。
        - **hop_length** (int) - 相邻滑动窗口之间的距离。
        - **win_length** (int) - 窗口和STFT过滤器的尺寸。
        - **normalized** (bool) - 控制是否返回规范化的STFT结果
        - **onesided** (bool) - 控制是否返回一半的结果，以避免实际输入的冗余。
        - **return_complex** (bool) - 若为True，返回一个复数Tensor。若为False，返回一个实数Tensor，
          且其具有额外的最后一维以表示实部和虚部。

    输入：
        - **x** (Tensor) - STFT的时间序列，必须是1-D Tensor或2-D Tensor。
        - **window** (Tensor) - 可选的窗口函数。

    输出：
        - **y** (Tensor) - 一个Tensor包含STFT的结果，其shape如上所述。

