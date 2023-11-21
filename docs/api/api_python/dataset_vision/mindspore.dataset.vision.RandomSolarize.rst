mindspore.dataset.vision.RandomSolarize
=======================================

.. py:class:: mindspore.dataset.vision.RandomSolarize(threshold=(0, 255))

    从给定阈值范围内随机选择一个子范围，对位于给定子范围内的像素，将其像素值设置为(255 - 原本像素值)。

    参数：
        - **threshold** (tuple, 可选) - 随机反转的阈值范围。默认值： ``(0, 255)`` 。 `threshold` 输入格式应该为 (min, max)，其中min和max是 [0, 255] 范围内的整数，并且min <= max，那么属于[min, max]这个区间的像素值会被反转。如果min与max相等，则反转所有大于等于 min(或max) 的像素值。

    异常：
        - **TypeError** - 当 `threshold` 的类型不为tuple。
        - **ValueError** - 当 `threshold` 取值不在[0, 255]范围内。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
