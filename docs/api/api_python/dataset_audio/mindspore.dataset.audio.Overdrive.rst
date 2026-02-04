mindspore.dataset.audio.Overdrive
=================================

.. py:class:: mindspore.dataset.audio.Overdrive(gain=20.0, color=20.0)

    给音频波形施加过载效果。

    接口实现方式类似于 `SoX库 <http://sox.sourceforge.net/sox.html>`_ 。

    参数：
        - **gain** (float, 可选) - 期望提升（或衰减）的音频增益，单位为dB，取值范围为[0, 100]。默认值： ``20.0`` 。
        - **color** (float, 可选) - 控制过载输出中偶次谐波成分的量，取值范围为[0, 100]。默认值： ``20.0`` 。

    异常：
        - **TypeError** - 当 `gain` 的类型不为float。
        - **ValueError** - 当 `gain` 取值不在[0, 100]范围内。
        - **TypeError** - 当 `color` 的类型不为float。
        - **ValueError** - 当 `color` 取值不在[0, 100]范围内。
        - **RuntimeError** - 当输入音频的shape不为<..., time>。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
