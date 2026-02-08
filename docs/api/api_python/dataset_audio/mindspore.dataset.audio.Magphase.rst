mindspore.dataset.audio.Magphase
================================

.. py:class:: mindspore.dataset.audio.Magphase(power=1.0)

    将shape为 :math:`(..., 2)` 的复值光谱图分离，输出幅度和相位。

    参数：
        - **power** (float, 可选) - 范数的幂，必须是非负的。默认值： ``1.0`` 。

    异常：
        - **RuntimeError** - 当输入音频的shape不为 :math:`(..., 2)` 。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
