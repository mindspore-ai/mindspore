mindspore.dataset.audio.Fade
============================

.. py:class:: mindspore.dataset.audio.Fade(fade_in_len=0, fade_out_len=0, fade_shape=FadeShape.LINEAR)

    向波形添加淡入和/或淡出。

    参数：
        - **fade_in_len** (int, 可选) - 淡入长度（时间帧），必须是非负。默认值：0。
        - **fade_out_len** (int, 可选) - 淡出长度（时间帧），必须是非负。默认值：0。
        - **fade_shape** (:class:`mindspore.dataset.audio.FadeShape` , 可选) - 淡入淡出形状，可以选择FadeShape提供的模式。默认值：FadeShape.LINEAR。

          - FadeShape.QUARTER_SINE，表示淡入淡出形状为四分之一正弦模式。
          - FadeShape.HALF_SINE，表示淡入形状为半正弦模式。
          - FadeShape.LINEAR，表示淡入淡出形状为线性模式。
          - FadeShape.LOGARITHMIC，表示淡入淡出形状为对数模式。
          - FadeShape.EXPONENTIAL，表示淡入淡出形状为指数模式。

    异常：
        - **RuntimeError** - 如果 `fade_in_len` 超过音频波形长度。
        - **RuntimeError** - 如果 `fade_out_len` 超过音频波形长度。
