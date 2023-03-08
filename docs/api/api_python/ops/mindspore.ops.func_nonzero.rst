mindspore.ops.nonzero
=====================

.. py:function:: mindspore.ops.nonzero(input)

    计算x中非零元素的下标。

    参数：
        - **input** (Tensor) - nonzero的输入，任意维度的Tensor，秩应大于1。其数据类型为数值型和布尔型。

    返回：
        Tensor，维度为2，类型为int64，表示输入中所有非零元素的下标。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `input` 的维度为0。