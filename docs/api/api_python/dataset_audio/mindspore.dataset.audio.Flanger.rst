mindspore.dataset.audio.Flanger
===============================

.. py:class:: mindspore.dataset.audio.Flanger(sample_rate, delay=0.0, depth=2.0, regen=0.0, width=71.0, speed=0.5, phase=25.0, modulation=Modulation.SINUSOIDAL, interpolation=Interpolation.LINEAR)

    对音频应用法兰盘效果。

    参数：
        - **sample_rate** (int) - 采样频率（单位：Hz），例如44100 (Hz)。
        - **delay** (float, 可选) - 延迟时间（毫秒），范围：[0, 30]。默认值：0.0。
        - **depth** (float, 可选) - 延迟深度（毫秒），范围：[0, 10]。默认值：2.0。
        - **regen** (float, 可选) - 反馈增益，单位为dB，范围：[-95, 95]。默认值：0.0。
        - **width** (float, 可选) - 延迟增益，单位为dB，范围：[0, 100]。默认值：71.0。
        - **speed** (float, 可选) - 调制速度，单位为Hz，范围：[0.1, 10]。默认值：0.5。
        - **phase** (float, 可选) - 多通道的相移百分比，范围：[0, 100]。默认值：25.0。
        - **modulation** (Modulation, 可选) - 指定调制模式。
          可以是Modulation.SINUSOIDAL或Modulation.TRIANGULAR之一。默认值：Modulation.SINUSOIDAL。
        - **interpolation** (Interpolation, 可选) - 指定插值模式。
          可以是Interpolation.LINEAR或Interpolation.QUADRATIC中的一种。默认值：Interpolation.LINEAR。
    