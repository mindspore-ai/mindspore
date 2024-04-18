mindspore.dataset.audio.DCShift
===============================

.. py:class:: mindspore.dataset.audio.DCShift(shift, limiter_gain=None)

    对输入音频波形施加直流移位。可以从音频中删除直流偏移（DC Offset）。

    参数：
        - **shift** (float) - 音频的移位量，值必须在[-2.0, 2.0]范围内。
        - **limiter_gain** (float, 可选) - 防止截断，仅在波峰生效。值应远小于1，如0.05或0.02。默认值： ``None`` ，将被设置为 `shift` 。

    异常：
        - **TypeError** - 如果 `shift` 不是float类型。
        - **ValueError** - 如果 `shift` 不在[-2.0, 2.0]范围内。
        - **TypeError** - 如果 `limiter_gain` 不是float类型。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
