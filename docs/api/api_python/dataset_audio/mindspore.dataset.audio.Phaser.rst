mindspore.dataset.audio.Phaser
==============================

.. py:class:: mindspore.dataset.audio.Phaser(sample_rate, gain_in=0.4, gain_out=0.74, delay_ms=3.0, decay=0.4, mod_speed=0.5, sinusoidal=True)

    给音频波形施加相位效果。

    接口实现方式类似于 `SoX库 <http://sox.sourceforge.net/sox.html>`_ 。

    参数：
        - **sample_rate** (int) - 波形的采样率，例如 ``44100`` (Hz)。
        - **gain_in** (float, 可选) - 期望输入提升（或衰减）的增益，单位为dB，取值范围为[0.0, 1.0]。默认值： ``0.4`` 。
        - **gain_out** (float, 可选) - 期望输出提升（或衰减）的增益，单位为dB，取值范围为[0.0, 1e9]。默认值： ``0.74`` 。
        - **delay_ms** (float, 可选) - 期望的时延，单位为毫秒，取值范围为[0.0, 5.0]。默认值： ``3.0`` 。
        - **decay** (float, 可选) - 期望的输入增益衰减系数，取值范围为[0.0, 0.99]。默认值： ``0.4`` 。
        - **mod_speed** (float, 可选) - 调制速率，单位为Hz，取值范围为[0.1, 2.0]。默认值： ``0.5`` 。
        - **sinusoidal** (bool, 可选) - 如果为 ``True`` ，将使用正弦调制（适用于多乐器场景）。
          如果为 ``False`` ，则使用三角调制（使单乐器获得更明显的相位效果）。默认值： ``True`` 。

    异常：
        - **TypeError** - 当 `sample_rate` 的类型不为int。
        - **TypeError** - 当 `gain_in` 的类型不为float。
        - **ValueError** - 当 `gain_in` 取值不在[0.0, 1.0]范围内。
        - **TypeError** - 当 `gain_out` 的类型不为float。
        - **ValueError** - 当 `gain_out` 取值不在[0.0, 1e9]范围内。
        - **TypeError** - 当 `delay_ms` 的类型不为float。
        - **ValueError** - 当 `delay_ms` 取值不在[0.0, 5.0]范围内。
        - **TypeError** - 当 `decay` 的类型不为float。
        - **ValueError** - 当 `decay` 取值不在[0.0, 0.99]范围内。
        - **TypeError** - 当 `mod_speed` 的类型不为float。
        - **ValueError** - 当 `mod_speed` 取值不在[0.1, 2.0]范围内。
        - **TypeError** - 当 `sinusoidal` 的类型不为bool。
        - **RuntimeError** - 当输入音频的shape不为<..., time>。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
