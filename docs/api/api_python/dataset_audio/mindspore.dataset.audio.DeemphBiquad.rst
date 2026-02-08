mindspore.dataset.audio.DeemphBiquad
====================================

.. py:class:: mindspore.dataset.audio.DeemphBiquad(sample_rate)

    给音频波形施加CD（IEC 60908）去重音（一种高音衰减搁置滤波器）效果。

    接口实现方式类似于 `SoX库 <http://sox.sourceforge.net/sox.html>`_ 。

    参数：
        - **sample_rate** (int) - 采样频率（单位：Hz），必须为 ``44100`` 或 ``48000``。

    异常：
        - **TypeError** - 当 `sample_rate` 的类型不为int。
        - **ValueError** - 当 `sample_rate` 不为 ``44100`` 或 ``48000`` 。
        - **RuntimeError** - 当输入音频的shape不为<..., time>。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
