mindspore.ops.standard_laplace
==============================

.. py:function:: mindspore.ops.standard_laplace(shape, seed=0, seed2=0)

    生成符合标准Laplace（mean=0, lambda=1）分布的随机数。
    其概率密度函数为：

    .. math::
        \text{f}(x) = \frac{1}{2}\exp(-|x|)

    参数：
        - **shape** (Union[tuple, Tensor]) - 待生成的Tensor的shape。当为tuple类型时，只支持常量值；当为Tensor类型时，支持动态Shape。
        - **seed** (int) - 算子层的随机种子，用于生成随机数。必须是非负的。默认值：0。
        - **seed2** (int) - 全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。必须是非负的。默认值：0。

    返回：
        Tensor，其shape为输入 `shape`。数据类型为float32。

    异常：
        - **TypeError** - `seed` 或 `seed2` 不是int。
        - **TypeError** - `shape` 不是tuple。
        - **ValueError** - `shape` 不是常量值。
