mindspore.nn.MSSSIM
====================

.. py:class:: mindspore.nn.MSSSIM(max_val=1.0, power_factors=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333), filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

    多尺度计算两个图像之间的结构相似性（SSIM）。

    基于Zhou Wang、Eero P.Simoncelli和Alan C.Bovik在2004年于Signals, Systems 和 Computers上发表的 `Multiscale structural similarity for image quality assessment<https://ieeexplore.ieee.org/document/1292216>`_ 。

    .. math::
        l(x,y)&=\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1}, C_1=(K_1L)^2.\\
        c(x,y)&=\frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2}, C_2=(K_2L)^2.\\
        s(x,y)&=\frac{\sigma_{xy}+C_3}{\sigma_x\sigma_y+C_3}, C_3=C_2/2.\\
        MSSSIM(x,y)&=l^\alpha_M*{\prod_{1\leq j\leq M} (c^\beta_j*s^\gamma_j)}.

    **参数：**

    - **max_val** (Union[int, float]) - 像素值的动态范围，即最大值和最小值之间的差值。（8bit灰度图像素为255）。默认值：1.0。
    - **power_factors** (Union[tuple, list]) - 权重列表，可循环获取权重值。默认值：(0.0448、0.2856、0.3001、0.2363、0.1333)。此处使用的默认值是由Wang等人在论文中提出。
    - **filter_size** (int) - 高斯过滤器的大小。默认值：11。
    - **filter_sigma** (float) - 高斯核的标准差。默认值：1.5。
    - **k1** (float) - 在亮度比较函数中，此常量用于生成 :math:`C_1` 。默认值：0.01。
    - **k2** (float) - 在对比度比较函数中，此常量用于生成 :math:`C_2` 。默认值：0.03。

    **输入：**

    - **img1** (Tensor) - 格式为'NCHW'的第一批图像。shape和数据类型必须与 `img2` 相同。
    - **img2** (Tensor) - 格式为'NCHW'的第二批图像。shape和数据类型必须与 `img1` 相同。

    **输出：**

    Tensor，值在[0, 1]范围内。它是一个shape为N的一维Tensor，其中N是 `img1` 的批次号。

    **异常：**

    - **TypeError** - 如果 `max_val` 既不是int也不是float。
    - **TypeError** - 如果 `power_factors` 既不是tuple也不是list。
    - **TypeError** - 如果 `k1` 、 `k2` 或 `filter_sigma` 不是float。
    - **TypeError** - 如果 `filter_size` 不是int。
    - **ValueError** - 如果 `max_val` 或 `filter_sigma` 小于或等于0。
    - **ValueError** - 如果 `filter_size` 小于0。
    - **ValueError** - 如果 `img1` 或 `img2` 的shape长度不等于4。