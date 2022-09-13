mindspore.dataset.audio.DeemphBiquad
====================================

.. py:class:: mindspore.dataset.audio.DeemphBiquad(sample_rate)

    为（..., time）形状的音频波形施加双极点去声滤波器。

    参数：
        - **sample_rate** (int) - 采样频率（单位：Hz），值必须为44100或48000。
    
    异常：
        - **RuntimeError** - 当输入音频的shape不为<..., time>。
