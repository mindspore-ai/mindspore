mindspore.nn.SSIM
==================

.. py:class:: mindspore.nn.SSIM(max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

    计算两个图像之间的结构相似性（SSIM）。

    SSIM是用来衡量两张图片相似性的指标。与PSNR一样，SSIM经常被用于图像质量的评估。SSIM是一个介于0和1之间的值，值越大，输出图像和未失真图像之间的差距越小，即图像质量越好。当两个图像完全相同时，SSIM=1。SSIM的实现请参考：Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004) `Image quality assessment: from error visibility to structural similarity <https://ieeexplore.ieee.org/document/1284395>`_ 

    .. math::
        l(x,y)&=\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1}, C_1=(K_1L)^2.\\
        c(x,y)&=\frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2}, C_2=(K_2L)^2.\\
        s(x,y)&=\frac{\sigma_{xy}+C_3}{\sigma_x\sigma_y+C_3}, C_3=C_2/2.\\
        SSIM(x,y)&=l*c*s\\&=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}.

    **参数：**

    - **max_val** (Union[int, float]) - 像素值的动态范围，即最大值和最小值之间的差值。（8bit灰度图像像素为255）。默认值：1.0。
    - **filter_size** (int) - 高斯滤波器的大小。该值必须大于等于1。默认值：11。
    - **filter_sigma** (float) - 高斯核的标准差。该值必须大于0。默认值：1.5。
    - **k1** (float) - 用于在亮度比较函数中生成 :math:`C_1` 的常量。默认值：0.01。
    - **k2** (float) - 用于在对比度比较函数中生成 :math:`C_2` 的常量。默认值：0.03。

    **输入：**

    - **img1** (Tensor)：格式为'NCHW'的第一批图像。shape和数据类型必须与img2相同。
    - **img2** (Tensor)：格式为'NCHW'的第二批图像。shape和数据类型必须与img1相同。

    **输出：**

    Tensor，数据类型与 `img1` 相同。shape为N的一维Tensor，其中N是 `img1` 的批次号。

    **异常：**

    - **TypeError** - `max_val` 既不是int也不是float。
    - **TypeError** - `k1` 、 `k2` 或 `filter_sigma` 不是float。
    - **TypeError** - `filter_size` 不是int。
    - **ValueError** - `max_val` 或 `filter_sigma` 小于或等于0。
    - **ValueError** - `filter_size` 小于0。