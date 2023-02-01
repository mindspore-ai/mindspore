mindspore.ops.BatchToSpace
===========================

.. py:class:: mindspore.ops.BatchToSpace(block_size, crops)

    将批处理数据重新排列到空间数据中。

    此操作将批处理维度N拆分为 `block_size` 大小的块（blocks），输出Tensor的维度N，即为拆分后对应的块数。输出Tensor的H、W维分别是原H、W维和 `block_size` 在给定裁剪量情况下的乘积。

    参数：
        - **block_size** (int) - 指定拆分的块大小，其值不能小于2。
        - **crops** (Union[list(int), tuple(int)]) - 指定H和W维度上的裁剪值，包含2个列表。每个列表包含2个整数。所有值都必须不小于0。crops[i]表示指定空间维度i的裁剪值，该维度对应于输入维度i+2。要求 :math:`input\_shape[i+2]*block\_size > crops[i][0]+crops[i][1]` 。

    输入：
        - **input_x** (Tensor) - 输入Tensor。必须是四维，第零维度（维度n）的大小必须可被 `block_size` 的乘积整除。数据类型为float16或float32。

    输出：
        Tensor，数据类型与输入Tensor相同。假设输入shape为 :math:`(n, c, h, w)` ，经过 `block_size` 和 `crops` 计算后。输出shape将为 :math:`(n', c', h', w')` ，其中

        - :math:`n' = n//(block\_size*block\_size)`
        - :math:`c' = c`
        - :math:`h' = h*block\_size-crops[0][0]-crops[0][1]`
        - :math:`w' = w*block\_size-crops[1][0]-crops[1][1]`

    异常：
        - **TypeError** - 如果 `block_size` 或 `crops` 的元素不是int。
        - **TypeError** - 如果 `crops` 既不是list也不是tuple。
        - **ValueError** - 如果 `block_size` 的值小于2。
