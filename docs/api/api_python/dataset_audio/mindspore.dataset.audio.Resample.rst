mindspore.dataset.audio.Resample
================================

.. py:class:: mindspore.dataset.audio.Resample(orig_freq=16000, new_freq=16000, resample_method=ResampleMethod.SINC_INTERPOLATION, lowpass_filter_width=6, rolloff=0.99, beta=None)

    将信号从一个频率重采样至另一个频率。支持指定重采样方法。

    参数：
        - **orig_freq** (float, 可选) - 信号的原始频率，必须为正。默认值：16000。
        - **new_freq** (float, 可选) - 预期的输出频率，必须为正。默认值：16000。
        - **resample_method** (:class:`mindspore.dataset.audio.ResampleMethod` , 可选) - 使用的重采样方法，可为ResampleMethod.SINC_INTERPOLATION
          或ResampleMethod.KAISER_WINDOW。默认值：ResampleMethod.SINC_INTERPOLATION。
        - **lowpass_filter_width** (int, 可选) - 控制滤波器的带宽，数值越大表示带宽越宽，但效率越低，必须为正。默认值：6。
        - **rolloff** (float, 可选) - 滤波器的滚降频率，是奈奎斯特公式的一部分。值越低越利于减少抗混叠，但同时也会减少一部分最高频率，
          取值范围为(0, 1]。默认值：0.99。
        - **beta** (float, 可选) - Kaiser窗的形状参数。默认值：None，将使用14.769656459379492。

    异常：
        - **TypeError** - 当 `orig_freq` 的类型不为float。
        - **ValueError** - 当 `orig_freq` 不为正数。
        - **TypeError** - 当 `new_freq` 的类型不为float。
        - **ValueError** - 当 `new_freq` 不为正数。
        - **TypeError** - 当 `resample_method` 的类型不为 :class:`mindspore.dataset.audio.ResampleMethod` 。
        - **TypeError** - 当 `lowpass_filter_width` 的类型不为int。
        - **ValueError** - 当 `lowpass_filter_width` 不为正数。
        - **TypeError** - 当 `rolloff` 的类型不为float。
        - **ValueError** - 当 `rolloff` 取值不在(0, 1]范围内。
        - **RuntimeError** - 当输入音频的shape不为<..., time>。
