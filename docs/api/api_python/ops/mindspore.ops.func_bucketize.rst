mindspore.ops.bucketize
==========================

.. py:function:: mindspore.ops.bucketize(input, boundaries, right=False)

    根据 `boundaries` 对 `input` 进行分桶，如果 `right` 为 ``False``，则左边界关闭，对于 `input` 中的每个元素 x，返回的索引满足以下规则:

    .. math::
        \begin{cases}
        boundaries[i-1] < x <= boundaries[i], & \text{if right} = False\\
        boundaries[i-1] <= x < boundaries[i], & \text{if right} = True\\
        \end{cases}

    参数：
        - **input** (Tensor) - 输入的Tensor。
        - **boundaries** (list) - 表示桶的边界值的有序列表。
        - **right** (bool, 可选) - 如果为 ``False``，则从边界获取输入中每个值的下限索引；如果为 ``True``，则改为获取上限索引。默认值：``False``。

    返回：
        Tensor，返回的索引值，shape与输入Tensor的shape相同，数据类型为int32。

    异常：
        - **TypeError** - `boundaries` 不是list。
        - **TypeError** - `input` 不是Tensor。
