mindspore.ops.unbind
========================

.. py:function:: mindspore.ops.unbind(input, dim=0)

    移除tensor的指定维度，返回一个沿该维度所有切片的元组。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (int) - 指定要移除的维度。默认 ``0`` 。

    返回：
        由多个tensor组成的tuple。
