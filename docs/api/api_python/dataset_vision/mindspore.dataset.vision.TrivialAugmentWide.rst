mindspore.dataset.vision.TrivialAugmentWide
===========================================

.. py:class:: mindspore.dataset.vision.TrivialAugmentWide(num_magnitude_bins=31, interpolation=Inter.NEAREST, fill_value=0)

    对输入图像应用TrivialAugmentWide数据增强方法。

    参考论文 `TrivialAugmentWide: Tuning-free Yet State-of-the-Art Data Augmentation <https://arxiv.org/abs/2103.10158>`_ 。

    只支持3通道RGB图像。

    参数：
        - **num_magnitude_bins** (int, 可选) - 不同变换幅值的个数，需不小于2 。默认值： ``31`` 。
        - **interpolation** (:class:`~.vision.Inter`, 可选) - 图像插值方法。可选值详见 :class:`mindspore.dataset.vision.Inter` 。
          默认值： ``Inter.NEAREST``。
        - **fill_value** (Union[int, tuple[int, int, int]], 可选) - 变换后超出原图外区域的像素填充值，取值需在 [0, 255] 范围内。默认值： ``0`` 。
          如果输入int，将用于填充所有 RGB 通道。
          如果输入tuple[int, int, int]，则分别用于填充R、G、B通道。

    异常：
        - **TypeError** - 如果 `num_magnitude_bins` 不是int类型。
        - **ValueError** - 如果 `num_magnitude_bins` 小于2。
        - **TypeError** - 如果 `interpolation` 不是 :class:`mindspore.dataset.vision.Inter` 类型。
        - **TypeError** - 如果 `fill_value` 不是int或tuple[int, int, int]类型。
        - **ValueError** - 如果 `fill_value` 取值不在[0, 255]范围。
        - **RuntimeError** - 如果输入图像的形状不是 <H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
