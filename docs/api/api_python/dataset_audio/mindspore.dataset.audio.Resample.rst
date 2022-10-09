mindspore.dataset.audio.Resample
================================

.. py:class:: mindspore.dataset.audio.Resample(orig_freq=16000, new_freq=16000, resample_method=ResampleMethod.SINC_INTERPOLATION, lowpass_filter_width=6, rolloff=0.99, beta=None)

    将音频波形从一个频率重新采样到另一个频率。必要时可以指定重采样方法。

    参数：
        - **orig_freq** (float, 可选) - 音频波形的原始频率，必须为正，默认值：16000。
        - **new_freq** (float, 可选) - 目标音频波形频率，必须为正，默认值：16000。
        - **resample_method** (ResampleMethod, 可选) - 重采样方法，可以是ResampleMethod.SINC_INTERPOLATION和ResampleMethod.KAISER_WINDOW。
          默认值=ResampleMethod.SINC_INTERPOLATION。
        - **lowpass_filter_width** (int, 可选) - 控制滤波器的宽度，越多意味着更清晰，但效率越低，必须为正。默认值：6。
        - **rolloff** (float, 可选) - 滤波器的滚降频率，作为Nyquist的一小部分。
          较低的值减少了抗锯齿，但也减少了一些最高频率，范围：(0, 1]。默认值：0.99。
        - **beta** (float, 可选) - 用于kaiser窗口的形状参数，默认值：None，将使用14.769656459379492。
