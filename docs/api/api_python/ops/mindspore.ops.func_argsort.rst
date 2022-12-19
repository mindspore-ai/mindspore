mindspore.ops.argsort
======================

.. py:function:: mindspore.ops.argsort(input_x, axis=-1, descending=False)

    返回输入Tensor沿轴按特定顺序排序索引。

    参数：
        - **input_x** (Tensor) - 待排序的输入Tensor。
        - **axis** (int) - 指定排序轴。默认值：-1。
        - **descending** (bool) - 控制输出顺序。如果 `descending` 为True，按照元素值升序排序，否则降顺排序。默认值：False。

    返回：
        Tensor，排序后输入Tensor的索引。数据类型为int32。

