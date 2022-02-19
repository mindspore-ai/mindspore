mindspore.dataset.vision.c_transforms.RandomCropDecodeResize
============================================================

.. py:class:: mindspore.dataset.vision.c_transforms.RandomCropDecodeResize(size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Inter.BILINEAR, max_attempts=10)

    "裁剪"、"解码"和"调整大小"的组合处理。该算子将在随机位置裁剪输入图像，以 RGB 模式对裁剪后的图像进行解码，并调整解码图像的大小。针对 JPEG 图像进行了优化, 可以获得更好的性能。

    **参数：**

    - **size**  (Union[int, sequence]) - 调整后图像的输出大小。大小值必须为正。
      如果 size 是整数，则返回大小为 (size, size) 的正方形裁剪。
      如果 size 是一个长度为 2 的序列，它应该是 (高度, 宽度)。
    - **scale**  (list, tuple, 可选) - 要裁剪的原始大小的各个大小的范围[min, max)，必须为非负数, 默认值：（0.08, 1.0）。
    - **ratio**  (list, tuple, 可选) - 宽高比的范围 [min, max) 裁剪，必须为非负数, 默认值：(3. / 4., 4. / 3.)。
    - **interpolation**  (Inter, 可选) - resize算子的图像插值模式, 默认值：Inter.BILINEAR。它可以是 [Inter.BILINEAR、Inter.NEAREST、Inter.BICUBIC、Inter.AREA、Inter.PILCUBIC] 中的任何一个。

      - **Inter.BILINEAR**: 双线性插值。
      - **Inter.NEAREST**: 最近邻插值。
      - **Inter.BICUBIC**: 双三次插值。
      - **Inter.AREA**: 像素区域插值。
      - **Inter.PILCUBIC**: Pillow库中实现的双三次插值，输入需为3通道格式。

    - **max_attempts**  (int, 可选) - 提出有效crop_area 的最大尝试次数, 默认值：10。
      如果超过，则回退到使用 center_crop 代替。 max_attempts 值必须为正数。

    **异常：**

    - **TypeError** - 如果 `size` 不是int或sequence类型或元素不为int类型。
    - **TypeError** - 如果 `scale` 不是tuple类型。
    - **TypeError** - 如果 `ratio` 不是tuple类型。
    - **TypeError** - 如果 `interpolation` 不是 :class:`mindspore.dataset.vision.Inter` 的类型。
    - **TypeError** - 如果 `max_attempts` 不是int类型。
    - **ValueError** - 如果 `size` 不是正数。
    - **ValueError** - 如果 `scale` 为负数。
    - **ValueError** - 如果 `ratio` 为负数。
    - **ValueError** - 如果 `max_attempts` 不是正数。
    - **RuntimeError** - 如果输入图像的shape不是一维序列。
