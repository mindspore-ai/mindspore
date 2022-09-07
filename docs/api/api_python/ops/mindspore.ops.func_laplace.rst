mindspore.ops.laplace
======================

.. py:function:: mindspore.ops.laplace(shape, mean, lambda_param, seed=None)

    根据拉普拉斯分布生成随机数。

    它的定义为：

    .. math::
        \text{f}(x;μ,λ) = \frac{1}{2λ}\exp(-\frac{|x-μ|}{λ}),

    参数：
        - **shape** (tuple) - 指定生成随机Tensor的shape。格式为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **mean** (Tensor) - 均值μ分布参数，指定峰的位置。数据类型为float32。
        - **lambda_param** (Tensor) - 用于控制此随机分布方差的参数。拉普拉斯分布的方差等于 `lambda_param` 平方的两倍。数据类型为float32。
        - **seed** (int) - 随机种子，用作生成随机数。默认值：0。

    返回：
        Tensor。输出shape应该是使用输入 `shape` 、 `mean` 和 `lambda_param`  广播后的shape。数据类型为float32。
