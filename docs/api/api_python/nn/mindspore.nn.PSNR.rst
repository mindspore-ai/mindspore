mindspore.nn.PSNR
==================

.. py:class:: mindspore.nn.PSNR(max_val=1.0)

    在批处理中计算两个图像的峰值信噪比（PSNR）。

    它为批处理中的每个图像生成PSNR值。假设输入为 :math:`I` 和 :math:`K` ，且shape都为 :math:`h*w` 。 :math:`MAX` 表示像素值的动态范围。

    .. math::
        MSE&=\frac{1}{hw}\sum\limits_{i=0}^{h-1}\sum\limits_{j=0}^{w-1}[I(i,j)-K(i,j)]^2\\
        PSNR&=10*log_{10}(\frac{MAX^2}{MSE})

    **参数：**

    - **max_val** (Union[int, float]) - 像素的动态范围（8位灰度图像为255）。该值必须大于0。默认值：1.0。

    **输入：**

    - **img1** (Tensor) - 格式为'NCHW'的第一批图像。shape和数据类型必须与 `img2` 相同。
    - **img2** (Tensor) - 格式为'NCHW'的第二批图像。shape和数据类型必须与 `img1` 相同。

    **输出：**

    Tensor，使用数据类型mindspore.float32。shape为N的一维Tensor，其中N是 `img1` 的批次大小。

    **异常：**

    - **TypeError** - `max_val` 既不是int也不是float。
    - **ValueError** - `max_val` 小于或等于0。
    - **ValueError** - `img1` 或 `img2` 的shape长度不等于4。