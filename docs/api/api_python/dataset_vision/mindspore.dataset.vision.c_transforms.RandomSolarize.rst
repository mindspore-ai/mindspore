mindspore.dataset.vision.c_transforms.RandomSolarize
====================================================

.. py:class:: mindspore.dataset.vision.c_transforms.RandomSolarize(threshold=(0, 255))

    从给定阈值范围内随机选择一个子范围，并对像素值位于给定子范围内的像素，将其值设置为（255 - 原本像素值）。

    **参数：**

    - **threshold** (tuple, optional) - 随机曝光的阈值范围，默认值：（0, 255）。`threshold` 输入格式应该为 (min, max)，其中 `min` 和 `max` 是 (0, 255) 范围内的整数，并且 min <= max。 如果 min=max，则反转所有高于 min(或max) 的像素值。
