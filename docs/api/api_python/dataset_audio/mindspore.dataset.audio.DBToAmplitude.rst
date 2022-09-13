mindspore.dataset.audio.DBToAmplitude
=====================================

.. py:class:: mindspore.dataset.audio.DBToAmplitude(ref, power)

    将音频波形从分贝转换为功率或振幅。

    参数：
        - **ref** (float) - 输出波形的缩放系数。
        - **power** (float) - 如果 `power` 等于1，则将分贝值转为功率；如果为0.5，则将分贝值转为振幅。
