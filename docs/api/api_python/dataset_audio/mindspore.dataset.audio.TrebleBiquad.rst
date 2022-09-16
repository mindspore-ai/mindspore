mindspore.dataset.audio.TrebleBiquad
====================================

.. py:class:: mindspore.dataset.audio.TrebleBiquad(sample_rate, gain, central_freq=3000, Q=0.707)

    对音频波形施加高音调控效果。类似于SoX实现。

    参数：
        - **sample_rate** (int) - 采样频率（单位：Hz），不能为零。
        - **gain** (float) - 期望提升（或衰减）的音频增益（单位：dB）。
        - **central_freq** (float) - 中心频率（单位：Hz），默认值：3000。
        - **Q** (float, 可选) - `品质因子 <https://zh.wikipedia.org/wiki/%E5%93%81%E8%B3%AA%E5%9B%A0%E5%AD%90>`_ ，能够反映带宽与采样频率和中心频率的关系，取值范围为(0, 1]，默认值：0.707。
