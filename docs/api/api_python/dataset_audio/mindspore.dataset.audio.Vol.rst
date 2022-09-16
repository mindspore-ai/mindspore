mindspore.dataset.audio.Vol
===========================

.. py:class:: mindspore.dataset.audio.Vol(gain, gain_type=GainType.AMPLITUDE)

    对音频波形施加放大或衰减。

    参数：
        - **gain** (float) - 增益调整的值。
          如果 `gain_type` = GainType.AMPLITUDE，则增益代表非负幅度比。
          如果 `gain_type` = GainType.POWER，则增益代表功率。
          如果 `gain_type` = GainType.DB，则增益代表分贝。
        - **gain_type** (GainType, 可选) - 增益类型，包含以下三个枚举值增益类型.GainType.AMPLITUDE、
          GainType.POWER和GainType.DB。默认值：GainType.AMPLITUDE。
