mindspore.dataset.audio.Magphase
================================

.. py:class:: mindspore.dataset.audio.Magphase(power=1.0)

    将shape为(..., 2)的复值光谱图分离，输出幅度和相位。

    参数：
        - **power** (float) - 范数的幂，必须是非负的。默认值：1.0。
    
    异常：
        - **RuntimeError** - 当输入音频的shape不为(..., 2)。
