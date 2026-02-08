mindspore.dataset.audio.Gain
============================

.. py:class:: mindspore.dataset.audio.Gain(gain_db=1.0)

    放大或衰减整个音频波形。

    参数：
        - **gain_db** (float, 可选) - 增益调整，单位为分贝（dB）。默认值： ``1.0`` 。

    异常：
        - **TypeError** - 当 `gain_db` 的类型不为float。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
