mindspore.dataset.audio.RiaaBiquad
==================================

.. py:class:: mindspore.dataset.audio.RiaaBiquad(sample_rate)

    对输入音频波形施加RIAA均衡。类似于SoX实现。

    参数：
        - **sample_rate** (int) - 波形的采样率，例如44100 (Hz)，只能是44100、48000、88200、96000中的一个。