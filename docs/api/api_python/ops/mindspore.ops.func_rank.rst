mindspore.ops.rank
===================

.. py:function:: mindspore.ops.rank(input_x)

    返回输入Tensor的秩。

    返回输入Tensor的秩，是一个0维的，数据类型为int32；Tensor的秩是确定Tensor中每个元素所需的索引数。

    参数：
        - **input_x** (Tensor) - Tensor的shape为 :math:`(x_1,x_2,...,x_R)` 。数据类型为数值型。

    返回：
        Tensor，表示输入的秩，是一个0维的，数据类型为int32，即 :math:`R` 。数据类型为int。

    异常：
        - **TypeError** - 如果 `input_x` 不是Tensor。
