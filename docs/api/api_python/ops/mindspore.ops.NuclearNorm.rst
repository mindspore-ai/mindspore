mindspore.ops.NuclearNorm
==========================

.. py:class:: mindspore.ops.NuclearNorm(dim=None, keepdim=False)

    返回给定Tensor的矩阵核范数。

    属性 `dim` 指定输入 `x` 的哪两个维度计算核范数。如果 `dim` 为 ``None`` ，则核规范将在输入所有维度上计算。
    因为核范数是矩阵的奇异值之和，此时的输入应该是二维的。也就是说，如果输入是二维，我们计算输入矩阵的核范数。此时， `dim` 应设为 ``None`` 。
    如果你设置了 `dim` ，它也需要在适当的范围内，否则将不生效。如果输入为三维及以上，属性 `dim` 是必需的。它指定了在哪两个输入维度计算核范数。
    
    根据 `dim` 列表，输入Tensor根据 `dim` 重新排列。 `dim` 指定的两个维度将被放在末尾，其他维度的顺序相对不变。对每个调整后的Tensor的切片执行SVD以获得奇异值，将所有奇异值求和即为获得核规范。

    参数：
        - **dim** (Union[list(int), tuple(int)]，可选) - 指定计算 `x` 矩阵核范数的哪两个维度，如果 `dim` 为 ``None`` ，则核规范将在输入所有维度上计算。 `dim` 的长度应该是2，其值应在此范围内：:math:`[-x\_rank,x\_rank)` 。x_rank是 `x` 的维度。dim[0]和dim[1]的值不能指向相同的维度。默认值： ``None`` 。
        - **keepdim** (bool，可选) - 输出Tensor是否保留维度。默认值： ``False`` 。

    输入：
        - **x** (Tensor) - 计算矩阵核范数的Tensor。 `x` 的维度应该大于等于2，数据类型为float32或者float64。

    输出：
        Tensor，如果 `keepdim` 为True，将输入中 `dim` 指定的维度变为1后即为输出shape，否则 `dim` 中指定维度被移除。数据类型与 `x` 一致。


    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型float32 或 float64。
        - **TypeError** - `dim` 的数据类型不是list(int) 或者 tuple(int)。
        - **TypeError** - `keepdim` 的数据类型不是bool。
        - **ValueError** - `x` 的维度小于2。
        - **ValueError** - 指定的 `dim` 的长度不等于2。
        - **ValueError** - 没有指定 `dim` 的时候， `x` 的维度不等于2。
        - **ValueError** - `dim[0]` 和 `dim[1]` 指向相同的维度。
        - **ValueError** - `dim[0]` 或者 `dim[1]` 超出范围：:math:`[-x\_rank, x\_rank)` ，其中x_rank 为 `x` 的维度。


