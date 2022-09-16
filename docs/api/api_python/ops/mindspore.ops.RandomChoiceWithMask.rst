mindspore.ops.RandomChoiceWithMask
=====================================

.. py:class:: mindspore.ops.RandomChoiceWithMask(count=256, seed=0, seed2=0)

    对输入进行随机取样，返回取样索引和掩码。

    输入必须是秩不小于1的Tensor。如果其秩大于等于2，则第一个维度指定样本数。索引Tensor和掩码Tensor有固定的shape。
    索引Tensor为取样的索引，掩码Tensor表示索引Tensor中的哪些元素取值为True。

    参数：
        - **count** (int) - 取样数量，必须大于0。默认值：256。
        - **seed** (int) - 随机种子。默认值：0。
        - **seed2** (int) - 随机种子2。默认值：0。

    输入：
        - **input_x** (Tensor[bool]) - 输入Tensor，bool类型。秩必须大于等于1且小于等于5。

    输出：
        两个Tensor，第一个为索引，另一个为掩码。

        - **index** (Tensor) - 2维Tensor。
        - **mask** (Tensor) - 1维Tensor。

    异常：
        - **TypeError** - `count` 不是int类型。
        - **TypeError** - `seed` 或 `seed2` 不是int类型。
        - **TypeError** - `input_x` 不是Tensor。
