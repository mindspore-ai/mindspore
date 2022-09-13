mindspore.dataset.audio.EqualizerBiquad
=======================================

.. py:class:: mindspore.dataset.audio.EqualizerBiquad(sample_rate, center_freq, gain, Q=0.707)

    给音频波形施加双二次均衡器滤波器。

    接口实现方式类似于 `SoX库 <http://sox.sourceforge.net/sox.html>`_ 。

    参数：
        - **sample_rate** (int) - 采样频率（单位：Hz），值不能为零。
        - **center_freq** (float) - 中心频率（单位：Hz）。
        - **gain** (float) - 期望提升（或衰减）的音频增益（单位：dB）。
        - **Q** (float, 可选) - `品质因子 <https://zh.wikipedia.org/wiki/%E5%93%81%E8%B3%AA%E5%9B%A0%E5%AD%90>`_ ，能够反映带宽与采样频率和中心频率的关系，取值范围为(0, 1]，默认值：0.707。
