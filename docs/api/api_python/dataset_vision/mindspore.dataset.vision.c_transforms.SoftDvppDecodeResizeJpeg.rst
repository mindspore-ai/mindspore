mindspore.dataset.vision.c_transforms.SoftDvppDecodeResizeJpeg
================================================================

.. py:class:: mindspore.dataset.vision.c_transforms.SoftDvppDecodeResizeJpeg(size)

    使用Ascend系列芯片DVPP模块的仿真算法对JPEG图像进行解码和缩放。

    建议在以下场景使用该算法：训练时不使用Ascend芯片的DVPP模块，推理时使用Ascend芯片的DVPP模块，推理的准确率低于训练的准确率； 并且输入图像大小应在 [32*32, 8192*8192] 范围内。 图像长度和宽度的缩小和放大倍数应在 [1/32, 16] 范围内。使用该算子只能输出具有均匀分辨率的图像，不支持奇数分辨率的输出。

    **参数：**

    - **size** (Union[int, sequence]) - 图像的输出大小。如果 `size` 是整数，将调整图像的较短边长度为 `size`，且保持图像的宽高比不变；若输入是2元素组成的序列，其输入格式需要是 (height, width) 。

    **异常：**

    - **TypeError** - 当 `size` 不是整型或不是整型序列。
    - **ValueError** - 当 `size` 不为正数。
    - **RuntimeError** - 如果输入的Tensor不是一个一维序列。
