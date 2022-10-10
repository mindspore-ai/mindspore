mindspore.ops.BoundingBoxEncode
================================

.. py:class:: mindspore.ops.BoundingBoxEncode(means=(0.0, 0.0, 0.0, 0.0), stds=(1.0, 1.0, 1.0, 1.0))

    编码边界框位置信息。

    算子的功能是计算预测边界框和真实边界框之间的偏移，并将此偏移作为损失变量。

    更多细节详见 :func:`mindspore.ops.boundingbox_encode`。