mindspore.dataset.audio.Phaser
==============================

.. py:class:: mindspore.dataset.audio.Phaser(sample_rate, gain_in=0.4, gain_out=0.74, delay_ms=3.0, decay=0.4, mod_speed=0.5, sinusoidal=True)

    对音频应用相位效果。

    参数：
        - **sample_rate** (int) - 波形的采样率，例如44100 (Hz)。
        - **gain_in** (float, 可选) - 期望提升（或衰减）所需输入增益，单位为dB。允许的值范围为[0, 1]，默认值：0.4。
        - **gain_out** (float, 可选) - 期望提升（或衰减）期望输出增益，单位为dB。允许的值范围为[0, 1e9]，默认值：0.74。
        - **delay_ms** (float, 可选) - 延迟数，以毫秒为单位。允许的值范围为[0, 5]，默认值：3.0。
        - **decay** (float, 可选) - 增益的期望衰减系数。允许的值范围为[0, 0.99]，默认值：0.4。
        - **mod_speed** (float, 可选) - 调制速度，单位为Hz。允许的值范围为[0.1, 2]，默认值：0.5。
        - **sinusoidal** (bool, 可选) - 如果为True，请使用正弦调制（对于多个乐器效果最好）。
          如果为False，则使用triangular modulation（使单个乐器具有更清晰的相位效果）。默认值：True。
  