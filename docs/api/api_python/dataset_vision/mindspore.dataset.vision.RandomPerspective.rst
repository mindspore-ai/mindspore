mindspore.dataset.vision.RandomPerspective
==========================================

.. py:class:: mindspore.dataset.vision.RandomPerspective(distortion_scale=0.5, prob=0.5, interpolation=Inter.BICUBIC)

    按照指定的概率对输入PIL图像进行透视变换。

    参数：
        - **distortion_scale** (float，可选) - 失真程度，取值范围为[0.0, 1.0]，默认值：0.5。
        - **prob** (float，可选) - 执行透视变换的概率，取值范围：[0.0, 1.0]。默认值：0.5。
        - **interpolation** (Inter，可选) - 插值方式，取值可为 Inter.BILINEAR、Inter.NEAREST 或 Inter.BICUBIC。默认值：Inter.BICUBIC。

          - **Inter.BILINEAR**：双线性插值。
          - **Inter.NEAREST**：最近邻插值。
          - **Inter.BICUBIC**：双三次插值。

    异常：
        - **TypeError** - 当 `distortion_scale` 的类型不为float。
        - **TypeError** - 当 `prob` 的类型不为float。
        - **TypeError** - 当 `interpolation` 的类型不为 :class:`mindspore.dataset.vision.Inter` 。
        - **ValueError** - 当 `distortion_scale` 取值不在[0.0, 1.0]范围内。
        - **ValueError** - 当 `prob` 取值不在[0.0, 1.0]范围内。
