mindspore.dataset.audio.MuLawDecoding
=====================================

.. py:class:: mindspore.dataset.audio.MuLawDecoding(quantization_channels=256)

    解码mu-law编码的信号，参考 `mu-law算法 <https://en.wikipedia.org/wiki/M-law_algorithm>`_ 。

    参数：
        - **quantization_channels** (int, 可选) - 通道数，必须为正数。默认值： ``256`` 。

    异常：
        - **TypeError** - 当 `quantization_channels` 的类型不为int。
        - **ValueError** - 当 `quantization_channels` 不为正数。
        - **RuntimeError** - 当输入音频的shape不为<..., time>。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
