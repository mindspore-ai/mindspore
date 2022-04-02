mindspore.dataset.vision.c_transforms.SoftDvppDecodeRandomCropResizeJpeg
========================================================================

.. py:class:: mindspore.dataset.vision.c_transforms.SoftDvppDecodeRandomCropResizeJpeg(size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), max_attempts=10)

    使用Ascend系列芯片DVPP模块的模拟算法对JPEG图像进行裁剪、解码和缩放。
    使用场景与数据增强算子 :class:`mindspore.dataset.vision.c_transforms.SoftDvppDecodeResizeJpeg` 一致。输入图像尺寸大小应在 [32*32, 8192*8192] 范围内。图像长度和宽度的缩小和放大倍数应在 [1/32, 16] 范围内。使用该算子只能输出具有均匀分辨率的图像，不支持奇数分辨率的输出。

    **参数：**

    - **size** (Union[int, Sequence[int]]) - 输出图像的尺寸大小。如果 `size` 是整型，则返回尺寸大小为 (size, size) 的正方形图像。如果 `size` 是一个长度为2的序列，则以2个元素分别为高和宽放缩至(高度, 宽度)大小。
    - **scale** (Union[list, tuple], 可选) - 裁剪子图的尺寸大小相对原图比例的随机选取范围，需要在[min, max)区间，默认值：(0.08, 1.0)。
    - **ratio** (Union[list, tuple], 可选) - 裁剪子图的宽高比的随机选取范围，需要在[min, max)区间，默认值：(3./4., 4./3.)。
    - **max_attempts** (int, 可选) - 生成随机裁剪位置的最大尝试次数，超过该次数时将使用中心裁剪， `max_attempts` 值必须为正数，默认值：10。

    **异常：**

    - **TypeError** - 当 `size` 不是int或不是Sequence[int]类型。
    - **TypeError** - 当 `scale` 不是tuple或list类型。
    - **TypeError** - 当 `ratio` 不是tuple或list类型。
    - **TypeError** - 当 `max_attempts` 不是int类型。
    - **ValueError** - 当 `size` 不为正数。
    - **ValueError** - 当 `scale` 是负数。
    - **ValueError** - 当 `ratio` 是负数。
    - **ValueError** - 当 `max_attempts` 不为正数。
    - **RuntimeError** - 如果输入的Tensor不是一个一维序列。
