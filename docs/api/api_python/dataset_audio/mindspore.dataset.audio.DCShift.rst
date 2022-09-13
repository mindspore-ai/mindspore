mindspore.dataset.audio.DCShift
===============================

.. py:class:: mindspore.dataset.audio.DCShift(shift, limiter_gain=None)

    对输入音频波形施加直流移位，可以从音频中删除直流偏移（DC Offset）。

    参数：
        - **ref** (float) - 音频的移位量，值必须在[-2.0, 2.0]范围内。
        - **limiter_gain** (float, 可选) - 防止截断，仅在波峰生效。值应远小于1，如0.05或0.02。
