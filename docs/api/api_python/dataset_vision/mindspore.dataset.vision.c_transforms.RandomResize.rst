mindspore.dataset.vision.c_transforms.RandomResize
==================================================

.. py:class:: mindspore.dataset.vision.c_transforms.RandomResize(size)

    使用随机选择的插值模式调整输入图像的大小。

    **参数：**

    - **size**  (Union[int, sequence]) - 调整后图像的输出大小。 大小值必须为正。
      如果 size 是整数，则图像的较小边缘将调整为具有相同图像纵横比的该值。
      如果 size 是一个长度为 2 的序列，它应该是 (高度, 宽度)。

    **异常：**

    - **TypeError** - 如果 `size` 不是int或sequence类型或元素不为int类型。
    - **ValueError** - 如果 `size` 不是正数。
    - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。
