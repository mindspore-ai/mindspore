mindspore.dataset.audio.Overdrive
=================================

.. py:class:: mindspore.dataset.audio.Overdrive(gain=20.0, color=20.0)

    对输入音频应用过载。

    参数：
        - **gain** (float, 可选) - 期望提升（或衰减）的音频增益（单位：dB），范围为[0, 100]。默认值：20.0。
        - **color** (float, 可选) - 控制过驱动输出中的偶次谐波内容量，范围为[0, 100]。默认值：20.0。
