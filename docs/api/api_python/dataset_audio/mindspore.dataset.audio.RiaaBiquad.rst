mindspore.dataset.audio.RiaaBiquad
==================================

.. py:class:: mindspore.dataset.audio.RiaaBiquad(sample_rate)

    对输入音频波形施加RIAA均衡。

    接口实现方式类似于 `SoX库 <http://sox.sourceforge.net/sox.html>`_ 。

    参数：
        - **sample_rate** (int) - 波形的采样率，单位为Hz。必须为 ``44100`` 、 ``48000`` 、 ``88200`` 、 ``96000`` 中的一个。

    异常：
        - **TypeError** - 当 `sample_rate` 的类型不为int。
        - **ValueError** - 当 `sample_rate` 不为 ``44100`` 、 ``48000`` 、 ``88200`` 、 ``96000`` 中的任何一个。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
