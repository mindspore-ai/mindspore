mindspore.dataset.audio.Flanger
===============================

.. py:class:: mindspore.dataset.audio.Flanger(sample_rate, delay=0.0, depth=2.0, regen=0.0, width=71.0, speed=0.5, phase=25.0, modulation=Modulation.SINUSOIDAL, interpolation=Interpolation.LINEAR)

    给音频施加镶边效果。

    接口实现方式类似于 `SoX库 <http://sox.sourceforge.net/sox.html>`_ 。

    参数：
        - **sample_rate** (int) - 波形采样频率，例如44100 (Hz)。
        - **delay** (float, 可选) - 期望的延迟时间，单位为毫秒，取值范围为[0, 30]。默认值：0.0。
        - **depth** (float, 可选) - 期望的延迟深度，单位为毫秒，取值范围为[0, 10]。默认值：2.0。
        - **regen** (float, 可选) - 期望的反馈增益，单位为dB，取值范围为[-95, 95]。默认值：0.0。
        - **width** (float, 可选) - 期望的延迟增益，单位为dB，取值范围为[0, 100]。默认值：71.0。
        - **speed** (float, 可选) - 调制速度，单位为Hz，取值范围为[0.1, 10]。默认值：0.5。
        - **phase** (float, 可选) - 各通道的相位偏移百分比，取值范围为[0, 100]。默认值：25.0。
        - **modulation** (:class:`mindspore.dataset.audio.Modulation` , 可选) - 调制方法，可为Modulation.SINUSOIDAL或Modulation.TRIANGULAR。
          默认值：Modulation.SINUSOIDAL。
        - **interpolation** (:class:`mindspore.dataset.audio.Interpolation` , 可选) - 插值方法，可为Interpolation.LINEAR或Interpolation.QUADRATIC。
          默认值：Interpolation.LINEAR。

    异常：
        - **TypeError** - 当 `sample_rate` 的类型不为int。
        - **ValueError** - 当 `sample_rate` 为零。
        - **TypeError** - 当 `delay` 的类型不为float。
        - **ValueError** - 当 `delay` 取值不在[0, 30]范围内。
        - **TypeError** - 当 `depth` 的类型不为float。
        - **ValueError** - 当 `depth` 取值不在[0, 10]范围内。
        - **TypeError** - 当 `regen` 的类型不为float。
        - **ValueError** - 当 `regen` 取值不在[-95, 95]范围内。
        - **TypeError** - 当 `width` 的类型不为float。
        - **ValueError** - 当 `width` 取值不在[0, 100]范围内。
        - **TypeError** - 当 `speed` 的类型不为float。
        - **ValueError** - 当 `speed` 取值不在[0.1, 10]范围内。
        - **TypeError** - 当 `phase` 的类型不为float。
        - **ValueError** - 当 `phase` 取值不在[0, 100]范围内。
        - **TypeError** - 当 `modulation` 的类型不为 :class:`mindspore.dataset.audio.Modulation` 。
        - **TypeError** - 当 `interpolation` 的类型不为 :class:`mindspore.dataset.audio.Interpolation` 。
        - **RuntimeError** - 当输入音频的shape不为<..., channel, time>。
