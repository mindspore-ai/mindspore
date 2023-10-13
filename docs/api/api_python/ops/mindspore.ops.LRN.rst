mindspore.ops.LRN
=================

.. py:class:: mindspore.ops.LRN(depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, norm_region="ACROSS_CHANNELS")

    局部响应归一化操作LRN(Local Response Normalization)。

    .. warning::
        LRN在Ascend平台已废弃，存在潜在精度问题。建议使用其他归一化方法，如 :class:`mindspore.ops.BatchNorm` 代替LRN。

    .. math::
        b_{c} = a_{c}\left(k + \frac{\alpha}{n}
        \sum_{c'=\max(0, c-n/2)}^{\min(N-1,c+n/2)}a_{c'}^2\right)^{-\beta}

    其中 :math:`a_{c}` 表示特征图中 :math:`c` 对应的具体像素值；
    :math:`n/2` 为参数 `depth_radius` ； :math:`k` 为参数 `bias` ；
    :math:`\alpha` 为参数 `alpha` ； :math:`\beta` 为参数 `beta` 。

    参数：
        - **depth_radius** (int) - 一维归一化窗口的半宽。默认值： ``5`` 。
        - **bias** (float) - 偏移量（通常为正以避免除零问题）。默认值： ``1.0`` 。
        - **alpha** (float) - 比例系数，通常为正。默认值： ``1.0`` 。
        - **beta** (float) - 指数。默认值： ``0.5`` 。
        - **norm_region** (str) - 指定归一化区域。可选值： ``"ACROSS_CHANNELS"`` 。默认值： ``"ACROSS_CHANNELS"`` 。

    输入：
        - **x** (Tensor) - 数据类型为float16或float32的四维Tensor。

    输出：
        Tensor，与 `x` 的shape和数据类型相同。

    异常：
        - **TypeError** - `depth_radius` 不是int类型。
        - **TypeError** - `bias` 、 `alpha` 或 `beta` 不是float类型。
        - **TypeError** - `norm_region` 不是str。
        - **TypeError** - `x` 不是Tensor。
