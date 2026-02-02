mindspore.dataset.vision.VideoDecoder
=====================================

.. py:class:: mindspore.dataset.vision.VideoDecoder(source)

    单流视频解码器。支持读取视频流元数据，并对H264、H265编码格式的视频提供抽帧能力。

    参数：
        - **source** (str) - 视频文件路径。

    异常：
        - **TypeError** - 如果 `source` 不是str类型。
        - **ValueError** - 如果 `source` 不存在或者权限被拒绝。

    .. py:method:: get_frames_at(indices)

        检索指定索引处的帧。

        参数：
            - **indices** (list[int]) - 要获取的帧索引列表。

        返回：
            numpy.ndarray，四维uint8视频数据。格式为 [T, H, W, C]。“T”是帧数，“H”是高度，“W”是宽度，“C”是RGB的通道。

        异常：
            - **TypeError** - 如果 `indices` 不是list类型。
            - **TypeError** - 如果 `indices` 的值不是int类型。
            - **ValueError** - 如果 `indices` 的值不在[0, 总帧数)范围内。

    .. py:method:: metadata
        :property:

        获取视频流的元数据。

        返回：
            dict，元数据的相关信息。
