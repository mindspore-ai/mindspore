mindspore.dataset.vision.TenCrop
================================

.. py:class:: mindspore.dataset.vision.TenCrop(size, use_vertical_flip=False)

    在输入PIL图像的中心与四个角处分别裁剪指定尺寸大小的子图，并将其翻转图一并返回。

    参数：
        - **size** (Union[int, Sequence[int, int]]) - 裁剪子图的尺寸大小。若输入int，则以该值为边长裁剪( `size` , `size` )尺寸大小的子图；若输入Sequence[int, int]，则以2个元素分别为高和宽裁剪子图。
        - **use_vertical_flip** (bool，可选) - 若为 ``True`` ，将对子图进行垂直翻转；否则进行水平翻转。默认值： ``False`` 。

    异常：        
        - **TypeError** - 当 `size` 的类型不为int或Sequence[int, int]。
        - **TypeError** - 当 `use_vertical_flip` 的类型不为bool。
        - **ValueError** - 当 `size` 不为正数。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
