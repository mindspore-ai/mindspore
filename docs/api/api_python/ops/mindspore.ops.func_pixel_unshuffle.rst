mindspore.ops.pixel_unshuffle
==============================

.. py:function:: mindspore.ops.pixel_unshuffle(x, downscale_factor)

    pixel_unshuffle函数。

    在多个输入平面组成的输入上面应用pixel_unshuffle算法。关于pixel_unshuffle算法详细介绍，请参考 `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network <https://arxiv.org/abs/1609.05158>`_ 。

    通常情况下，`x` shape :math:`(*, C, H \times r, W \times r)` ，输出shape :math:`(*, C \times r^2, H, W)` 。`r` 是缩小因子。 `*` 是大于等于0的维度。

    参数：
        - **x** (Tensor) - Tensor，shape为 :math:`(*, C, H \times r, W \times r)` 。 `x` 的维度需要大于2，并且倒数第一和倒数第二维length可以被 `downscale_factor` 整除。
        - **downscale_factor** (int) - 减小空间分辨率的因子，是正整数。

    返回：
        - **output** (Tensor) - Tensor，shape为 :math:`(*, C \times r^2, H, W)` 。

    异常：
        - **ValueError** - `downscale_factor` 不是正整数。
        - **ValueError** - `x` 倒数第一和倒数第二维度的length不能被 `downscale_factor` 整除。
        - **TypeError** - `x` 维度小于3。
