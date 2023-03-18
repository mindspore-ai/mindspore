mindspore.ops.RandomChoiceWithMask
=====================================

.. py:class:: mindspore.ops.RandomChoiceWithMask(count=256, seed=0, seed2=0)

    对输入进行随机取样，返回取样索引和掩码。

    更多参考详见 :func:`mindspore.ops.choice_with_mask`。

    参数：
        - **count** (int，可选) - 取样数量，必须大于0。默认值：256。
        - **seed** (int, 可选) - 随机种子。默认值：None。
        - **seed2** (int, 可选) - 避免冲突的另一随机种子。默认值：0。

    输入：
        - **input_x** (Tensor[bool]) - 输入Tensor，bool类型。秩必须大于等于1且小于等于5。

    输出：
        两个Tensor，第一个为索引，另一个为掩码。

        - **index** (Tensor) - 二维Tensor。
        - **mask** (Tensor) - 一维Tensor。
