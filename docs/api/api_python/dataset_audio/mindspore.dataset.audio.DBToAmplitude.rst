mindspore.dataset.audio.DBToAmplitude
=====================================

.. py:class:: mindspore.dataset.audio.DBToAmplitude(ref, power)

    将音频波形从分贝转换为功率或振幅。

    参数：
        - **ref** (float) - 输出波形的缩放系数。
        - **power** (float) - 如果 `power` 等于 ``1`` ，则将分贝值转为功率；如果为 ``0.5`` ，则将分贝值转为振幅。

    异常：
        - **TypeError** - 如果 `ref` 不是float类型。
        - **TypeError** - 如果 `power` 不是float类型。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
