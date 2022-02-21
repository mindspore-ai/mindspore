mindspore.dataset.vision.c_transforms.SoftDvppDecodeRandomCropResizeJpeg
========================================================================

.. py:class:: mindspore.dataset.vision.c_transforms.SoftDvppDecodeRandomCropResizeJpeg(size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), max_attempts=10)

    使用Ascend系列芯片DVPP模块的模拟算法组合了 `Crop` 、 `Decode` 和 `Resize` 。
    使用场景与数据增强算子 `SoftDvppDecodeResizeJpeg` 一致。 输入图像大小应在 [32*32, 8192*8192] 范围内。图像长度和宽度的缩小和放大倍数应在 [1/32, 16] 范围内。使用该算子只能输出具有均匀分辨率的图像，不支持奇数分辨率的输出。

    **参数：**

    - **size** (Union[int, sequence]) - 输出图像的大小。如果 `size` 是整数，则返回大小为 (size, size) 的正方形图像。 如果 `size` 是一个长度为2的序列，其输入格式应该为 (height, width)。
    - **scale** (list, tuple, optional) - 裁剪子图的大小相对原图比例的随机选取范围，需要在[min, max)区间，默认值：(0.08, 1.0)。
    - **ratio** (list, tuple, optional) - 裁剪子图的宽高比的随机选取范围，需要在[min, max)区间，默认值：(3./4., 4./3.)。
    - **max_attempts** (int, optional) - 生成随机裁剪位置的最大尝试次数，超过该次数时将使用中心裁剪，默认值：10。

    **异常：**

    - **TypeError** - 当 `size` 不是整型或不是整型序列。
    - **TypeError** - 当 `scale` 不是元组类型。
    - **TypeError** - 当 `ratio` 不是元组类型。
    - **TypeError** - 当 `max_attempts` 不是整型。
    - **ValueError** - 当 `size` 不为正数。
    - **ValueError** - 当 `scale` 是负数。
    - **ValueError** - 当 `ratio` 是负数。
    - **ValueError** - 当 `max_attempts` 不为正数。
    - **RuntimeError** - 如果输入的Tensor不是一个一维序列。
