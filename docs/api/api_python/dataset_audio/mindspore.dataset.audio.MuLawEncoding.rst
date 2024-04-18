mindspore.dataset.audio.MuLawEncoding
=====================================

.. py:class:: mindspore.dataset.audio.MuLawEncoding(quantization_channels=256)

    基于mu-law压缩的信号编码。

    参数：
        - **quantization_channels** (int, 可选) - 通道数，必须为正数。默认值： ``256`` 。

    异常：
        - **TypeError** - 当 `quantization_channels` 的类型不为int。
        - **ValueError** - 当 `quantization_channels` 不为正数。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
