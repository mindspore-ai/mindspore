mindspore.ops.laplace
======================

.. py:function:: mindspore.ops.laplace(shape, mean, lambda_param, seed=None)

    根据拉普拉斯分布生成随机数。

    支持广播。

    .. math::
        \text{f}(x;μ,λ) = \frac{1}{2λ}\exp(-\frac{|x-μ|}{λ}),

    其中 :math:`μ` 为均值，代表了 `mean`， :math:`λ` 为尺度，代表了 `lambda_param`。

    .. warning::
        Ascend后端不支持随机数重现功能， `seed` 参数不起作用。

    参数：
        - **shape** (tuple) - 生成随机数的shape。
        - **mean** (Tensor) - 分布的均值。
        - **lambda_param** (Tensor) - 控制分布方差。拉普拉斯分布的方差等于 `lambda_param` 平方的两倍。
        - **seed** (int，可选) - 随机种子。默认 ``None`` 表示使用0。

    返回：
        Tensor
