mindspore.dataset.vision.c_transforms.RandomResizedCropWithBBox
================================================================

.. py:class:: mindspore.dataset.vision.c_transforms.RandomResizedCropWithBBox(size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Inter.BILINEAR, max_attempts=10)

    对输入图像进行随机裁剪且随机调整纵横比，并将处理后的图像调整为指定的大小，并相应地调整边界框。

    **参数：**

    - **size** (Union[int, sequence]) - 图像的输出大小。 如果 `size` 是整数，则返回大小为 (size, size) 的正方形图像。 如果 `size` 是一个长度为2的序列，其输入格式应该为 (height, width)。
    - **scale** (list, tuple, optional) - 裁剪子图的大小相对原图比例的随机选取范围，需要在[min, max)区间，默认值：(0.08, 1.0)。
    - **ratio** (list, tuple, optional) - 裁剪子图的宽高比的随机选取范围，需要在[min, max)区间，默认值：(3./4., 4./3.)。
    - **interpolation** (Inter, optional) - 调整大小采用的的图像插值模式。它可以是 [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC] 中的任何一个，默认值：Inter.BILINEAR。

      - Inter.BILINEAR，双线性插值。
      - Inter.NEAREST，最近邻插值。
      - Inter.BICUBIC，双三次插值。

     - **max_attempts** (int, optional):  生成随机裁剪位置的最大尝试次数，超过该次数时将使用中心裁剪，默认值：10。
