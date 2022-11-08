mindspore.ops.StandardLaplace
=============================

.. py:class:: mindspore.ops.StandardLaplace(seed=0, seed2=0)

    生成符合标准Laplace（mean=0, lambda=1）分布的随机数。
    其概率密度函数为：

    .. math::
        \text{f}(x) = \frac{1}{2}\exp(-|x|)

    参数：    
        - **seed** (int) - 算子层的随机种子，用于生成随机数。必须是非负的。默认值：0。
        - **seed2** (int) - 全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。必须是非负的。默认值：0。

    输入：    
        - **shape** (Union[tuple, Tensor]) - 待生成的Tensor的shape。当为tuple类型时，只支持常量值；当为Tensor类型时，支持动态Shape。

    输出：    
        Tensor。shape为输入 `shape` 。数据类型支持float32。

    异常：    
        - **TypeError** - `seed` 或 `seed2` 不是int。
        - **TypeError** - `shape` 既不是tuple，也不是Tensor。
        - **ValueError** - `seed` 或 `seed2` 不是非负的int。
        - **ValueError** - `shape` 为tuple时，包含非正的元素。
        - **ValueError** - `shape` 为秩不等于1的Tensor。
