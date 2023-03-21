mindspore.dataset.audio.AmplitudeToDB
=====================================

.. py:class:: mindspore.dataset.audio.AmplitudeToDB(stype=ScaleType.POWER, ref_value=1.0, amin=1e-10, top_db=80.0)

    将输入音频从振幅/功率标度转换为分贝标度。

    .. note:: 待处理音频shape需为<..., freq, time>。

    参数：
        - **stype** (:class:`mindspore.dataset.audio.ScaleType` , 可选) - 输入音频的原始标度，取值可为ScaleType.MAGNITUDE或ScaleType.POWER。默认值：ScaleType.POWER。
        - **ref_value** (float, 可选) - 系数参考值。默认值：1.0，用于计算分贝系数 `db_multiplier` ，公式为
          :math:`\text{db_multiplier} = Log10(max(\text{ref_value}, amin))` 。

        - **amin** (float, 可选) - 波形取值下界，低于该值的波形将会被裁切，取值必须大于0。默认值：1e-10。
        - **top_db** (float, 可选) - 最小截止分贝值，取值为非负数。默认值：80.0。

    异常：
        - **TypeError** - 当 `stype` 的类型不为 :class:`mindspore.dataset.audio.ScaleType` 。
        - **TypeError** - 当 `ref_value` 的类型不为float。
        - **ValueError** - 当 `ref_value` 不为正数。
        - **TypeError** - 当 `amin` 的类型不为float。
        - **ValueError** - 当 `amin` 不为正数。
        - **TypeError** - 当 `top_db` 的类型不为float。
        - **ValueError** - 当 `top_db` 不为正数。
        - **RuntimeError** - 当输入音频的shape不为<..., freq, time>。
