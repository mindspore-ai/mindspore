mindspore.dataset.vision.CenterCrop
===================================

.. py:class:: mindspore.dataset.vision.CenterCrop(size)

    对输入图像应用中心区域裁剪。如果输入图像尺寸小于输出尺寸，则在裁剪前对输入图像边界填充0像素。

    参数：
        - **size** (Union[int, sequence]) - 裁剪区域尺寸大小。
          如果 `size` 是整数，则返回一个裁剪尺寸大小为 (size, size) 的正方形。
          如果 `size` 是一个长度为 2 的序列，则以2个元素分别为高和宽放缩至(高度, 宽度)大小。
          值必须大于 0。

    异常：
        - **TypeError** - 如果 `size` 不是int或sequence类型。
        - **ValueError** - 如果 `size` 小于或等于 0。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <..., H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
