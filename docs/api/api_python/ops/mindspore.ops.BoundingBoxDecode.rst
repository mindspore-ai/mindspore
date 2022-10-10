mindspore.ops.BoundingBoxDecode
===============================

.. py:class:: mindspore.ops.BoundingBoxDecode(max_shape, means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0), wh_ratio_clip=0.016)

    解码边界框位置信息。

    算子的功能是计算偏移量，此算子将偏移量转换为Bbox，用于在后续图像中标记目标等。

    更多细节详见 :func:`mindspore.ops.boundingbox_decode`。
