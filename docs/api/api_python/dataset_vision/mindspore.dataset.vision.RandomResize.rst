mindspore.dataset.vision.RandomResize
=====================================

.. py:class:: mindspore.dataset.vision.RandomResize(size)

    使用随机选择的 :class:`mindspore.dataset.vision.Inter` 插值方式，调整输入图像的尺寸大小。

    参数：
        - **size** (Union[int, Sequence[int]]) - 调整后图像的输出尺寸大小。值必须为正。若输入整型，将调整图像的较短边长度为 `size` ，且保持图像的宽高比不变；若输入2元素序列，则以2个元素分别为高和宽放缩至(高度, 宽度)大小。

    异常：
        - **TypeError** - 如果 `size` 不是int或Sequence[int]类型。
        - **ValueError** - 如果 `size` 不是正数。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
