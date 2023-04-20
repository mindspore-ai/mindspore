mindspore.ops.choice_with_mask
=====================================

.. py:function:: mindspore.ops.choice_with_mask(input_x, count=256, seed=None)

    对输入进行随机取样，返回取样索引和掩码。

    输入必须是维度不小于1的Tensor。如果其维度大于等于2，则第一个维度指定样本数。
    返回的索引Tensor为非空样本值的索引，掩码Tensor说明索引Tensor中的哪些元素是有效的。

    参数：
        - **input_x** (Tensor[bool]) - 输入Tensor，bool类型。秩必须大于等于1且小于等于5。
        - **count** (int, 可选) - 取样数量，必须大于0。默认值： ``256`` 。
        - **seed** (int, 可选) - 随机种子。默认值： ``None`` 。

    返回：
        两个Tensor，第一个为索引，另一个为掩码。

        - **index** (Tensor) - 二维Tensor。
        - **mask** (Tensor) - 一维Tensor。

    异常：
        - **TypeError** - `count` 不是int类型。
        - **TypeError** - `seed` 不是int类型。
        - **TypeError** - `input_x` 不是Tensor。
