mindspore.dataset.vision.RandAugment
====================================

.. py:class:: mindspore.dataset.vision.RandAugment(num_ops=2, magnitude=9, num_magnitude_bins=32, interpolation=Inter.NEAREST, fill_value=0)

    对输入图像应用RandAugment数据增强方法。

    参考论文 `RandAugment: Learning Augmentation Strategies from Data <https://arxiv.org/pdf/1909.13719.pdf>`_ 。

    只支持3通道RGB图像。

    参数：
        - **num_ops** (int, 可选) - 顺序执行的数据增强变换个数。默认值：2。
        - **magnitude** (int, 可选) - 所有变换的幅值，需小于 `num_magnitude_bins` 。默认值：9。
        - **num_magnitude_bins** (int, 可选) - 不同变换幅值的个数，需不小于2 。默认值：31。
        - **interpolation** (Inter, 可选) - 图像插值方式。默认值：Inter.NEAREST。
          可为 Inter.NEAREST、Inter.BILINEAR、Inter.BICUBIC、Inter.AREA]。

          - **Inter.NEAREST** - 最近邻插值。
          - **Inter.BILINEAR** - 双线性插值。
          - **Inter.BICUBIC** - 双三次插值。
          - **Inter.AREA** - 像素区域插值。

        - **fill_value** (Union[int, tuple[int, int, int]], 可选) - 变换后超出原图外区域的像素填充值，取值需在 [0, 255] 范围内。默认值：0。
          如果输入int，将用于填充所有 RGB 通道。
          如果输入tuple[int, int, int]，则分别用于填充R、G、B通道。

    异常：
        - **TypeError** - 如果 `num_ops` 不是int类型。
        - **ValueError** - 如果 `num_ops` 为负数。
        - **TypeError** - 如果 `magnitude` 不是int类型。
        - **ValueError** - 如果 `magnitude` 非正数。
        - **TypeError** - 如果 `num_magnitude_bins` 不是int类型。
        - **ValueError** - 如果 `num_magnitude_bins` 小于2。
        - **TypeError** - 如果 `interpolation` 不是 :class:`mindspore.dataset.vision.Inter` 类型。
        - **TypeError** - 如果 `fill_value` 不是int或tuple[int, int, int]类型。
        - **ValueError** - 如果 `fill_value` 取值不在[0, 255]范围。
        - **RuntimeError** - 如果输入图像的形状不是 <H, W, C>。
