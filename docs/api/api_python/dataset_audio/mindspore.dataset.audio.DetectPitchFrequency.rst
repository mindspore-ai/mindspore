mindspore.dataset.audio.DetectPitchFrequency
============================================

.. py:class:: mindspore.dataset.audio.DetectPitchFrequency(sample_rate, frame_time=0.01, win_length=30, freq_low=85, freq_high=3400)

    检测音调频率，基于归一化互相关函数和中位平滑来实现。

    参数：
        - **sample_rate** (int) - 采样频率（单位：Hz），值必须为44100或48000。
        - **frame_time** (float, 可选) - 帧的持续时间，值必须大于零。默认值：0.01。
        - **win_length** (int, 可选) - 中位平滑的窗口长度（以帧数为单位），该值必须大于零。默认值：30。
        - **freq_low** (int, 可选) - 可检测的最低频率（Hz），该值必须大于零。默认值：85。
        - **freq_high** (int, 可选) - 可检测的最高频率（Hz），该值必须大于零。默认值：3400。
