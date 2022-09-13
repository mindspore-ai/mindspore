mindspore.dataset.audio.Fade
============================

.. py:class:: mindspore.dataset.audio.Fade(fade_in_len=0, fade_out_len=0, fade_shape=FadeShape.LINEAR)

    向波形添加淡入和/或淡出。

    参数：
        - **fade_in_len** (int, 可选) - 淡入长度（时间帧），必须是非负。默认值：0。
        - **fade_out_len** (int, 可选) - 淡出长度（时间帧），必须是非负。默认值：0。
        - **fade_shape** (FadeShape, 可选) - 淡入淡出形状，可以是FadeShape.QUARTER_SINE、FadeShape.HALF_SINE、
          FadeShape.LINEAR、FadeShape.LOGARITHMIC或FadeShape.EXPONENTIAL中的一个。默认值：FadeShape.LINEAR。

    异常：
        - **RuntimeError** - 如果 `fade_in_len` 超过音频波形长度。
        - **RuntimeError** - 如果 `fade_out_len` 超过音频波形长度。
