mindspore.ops.pixel_shuffle
============================

.. py:function:: mindspore.ops.pixel_shuffle(x, upscale_factor)

    pixel_shuffle函数。

    在多个输入平面组成的输入上面应用pixel_shuffle算法。在平面上应用高效亚像素卷积，步长为 :math:`1/r` 。关于pixel_shuffle算法详细介绍，请参考 `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`_ 。

    通常情况下，`x` shape :math:`(*, C \times r^2, H, W)` ，输出shape :math:`(*, C, H \times r, W \times r)` 。`r` 是缩小因子。 `*` 是大于等于0的维度。

    参数：
        - **x** (Tensor) - Tensor，shape为 :math:`(*, C \times r^2, H, W)` 。 `x` 的维度需要大于2，并且倒数第三维length可以被 `upscale_factor` 的平方整除。
        - **upscale_factor** (int) - 增加空间分辨率的因子，是正整数。。

    返回：
        - **output** (Tensor) - Tensor，shape为 :math:`(*, C, H \times r, W \times r)` 。

    异常：
        - **ValueError** - `upscale_factor` 不是正整数。
        - **ValueError** - `x` 倒数第三维度的length不能被 `upscale_factor` 的平方整除。
        - **TypeError** - `x` 维度小于3。
