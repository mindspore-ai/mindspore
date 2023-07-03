mindspore.ops.StandardLaplace
=============================

.. py:class:: mindspore.ops.StandardLaplace(seed=0, seed2=0)

    生成符合标准Laplace（mean=0, lambda=1）分布的随机数。
    其概率密度函数为：

    .. math::
        \text{f}(x) = \frac{1}{2}\exp(-|x|)

    .. note::
        - 随机种子：通过一些复杂的数学算法，可以得到一组有规律的随机数，而随机种子就是这个随机数的初始值。随机种子相同，得到的随机数就不会改变。
        - 全局的随机种子和算子层的随机种子都没设置或都设置为0：完全随机。
        - 全局的随机种子设置了，算子层的随机种子未设置：采用全局的随机种子和0拼接。
        - 全局的随机种子未设置，算子层的随机种子设置了：使用0和算子层的随机种子拼接。
        - 全局的随机种子和算子层的随机种子都设置了：全局的随机种子和算子层的随机种子拼接。

    参数：
        - **seed** (int，可选) - 算子层的随机种子，用于生成随机数。必须是非负的。默认值： ``0`` 。
        - **seed2** (int，可选) - 全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。必须是非负的。默认值： ``0`` 。

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
