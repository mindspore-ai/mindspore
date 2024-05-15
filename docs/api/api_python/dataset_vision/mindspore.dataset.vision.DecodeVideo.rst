mindspore.dataset.vision.DecodeVideo
====================================

.. py:class:: mindspore.dataset.vision.DecodeVideo()

    将输入的视频原始字节解码为视频、音频。

    支持的视频格式有AVI、H264、H265、MOV、MP4和WMV。

    异常：
        - **RuntimeError** - 如果输入不是一维序列。
        - **RuntimeError** - 如果输入数据的数据类型不是 uint8。
        - **RuntimeError** - 如果输入数据为空。
