mindspore.ops.argsort
======================

.. py:function:: mindspore.ops.argsort(input, axis=-1, descending=False)

    按指定顺序对输入Tensor沿给定维度进行排序，并返回排序后的索引。

    参数：
        - **input** (Tensor) - 待排序的输入Tensor。
        - **axis** (int) - 指定排序的轴。默认值：``-1``，表示指定最后一维。当前Ascend后端只支持对最后一维进行排序。
        - **descending** (bool) - 输出顺序。如果 `descending` 为 ``True`` ，按照元素值降序排序，否则升序排序。默认值： ``False`` 。

    返回：
        Tensor，排序后输入Tensor的索引。数据类型为int32。

