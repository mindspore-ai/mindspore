mindspore.dataset.audio.Vol
===========================

.. py:class:: mindspore.dataset.audio.Vol(gain, gain_type=GainType.AMPLITUDE)

    调整波形的音量。

    参数：
        - **gain** (float) - 提升（或衰减）的增益。
          如果 `gain_type` 为 GainType.AMPLITUDE，应为一个非负的幅度比。
          如果 `gain_type` 为 GainType.POWER，应为一个功率（电压的平方）。
          如果 `gain_type` 为 GainType.DB，应以分贝为单位。
        - **gain_type** (:class:`mindspore.dataset.audio.GainType` , 可选) - 增益的类型，可为GainType.AMPLITUDE、
          GainType.POWER或GainType.DB。默认值：GainType.AMPLITUDE。

    异常：
        - **TypeError** - 当 `gain` 的类型不为float。
        - **TypeError** - 当 `gain_type` 的类型不为 :class:`mindspore.dataset.audio.GainType` 。
        - **ValueError** - 当 `gain_type` 为 GainType.AMPLITUDE 时，`gain` 为负数。
        - **ValueError** - 当 `gain_type` 为 GainType.POWER 时，`gain` 不为正数。
        - **RuntimeError** - 当输入音频的shape不为<..., time>。
