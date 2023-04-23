mindspore.ops.standard_laplace
==============================

.. py:function:: mindspore.ops.standard_laplace(shape, seed=None)

    生成符合标准Laplace（mean=0, lambda=1）分布的随机数。
    其概率密度函数为：

    .. math::
        \text{f}(x) = \frac{1}{2}\exp(-|x|)

    参数：
        - **shape** (Union[tuple, Tensor]) - 待生成的Tensor的shape。当为tuple类型时，只支持常量值；当为Tensor类型时，支持动态Shape。
        - **seed** (int, 可选) - 算子层的随机种子，用于生成随机数。必须是非负的。默认值： ``None`` 。

    返回：
        Tensor，其shape为输入 `shape`。数据类型为float32。

    异常：
        - **TypeError** - `shape` 既不是tuple，也不是Tensor。
        - **ValueError** - `shape` 为tuple时，包含非正的元素。
        - **ValueError** - `shape` 为秩不等于1的Tensor。
