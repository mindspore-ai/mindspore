mindspore.dataset.vision.AdjustGamma
====================================

.. py:class:: mindspore.dataset.vision.AdjustGamma(gamma, gain=1)

    对输入图像应用伽马校正。输入图片shape应该为 <..., H, W, C>或<H, W>。

    .. math::
        I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}

    更多详细信息，请参见 `Gamma矫正 <https://en.wikipedia.org/wiki/Gamma_correction>`_ 。

    参数：
        - **gamma** (float) - 非负实数。输出图像像素值与输入图像像素值呈指数相关。 `gamma` 大于1使阴影更暗，而 `gamma` 小于1使黑暗区域更亮。
        - **gain** (float, 可选) - 常数乘数。默认值：1.0。

    异常：
        - **TypeError** - 如果 `gain` 不是浮点类型。
        - **TypeError** - 如果 `gamma` 不是浮点类型。
        - **ValueError** - 如果 `gamma` 小于0。
        - **RuntimeError** - 如果给定的张量形状不是<H, W>或<..., H, W, C>。
