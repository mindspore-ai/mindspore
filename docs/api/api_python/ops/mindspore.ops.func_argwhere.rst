mindspore.ops.argwhere
======================

.. py:function:: mindspore.ops.argwhere(input)

    返回一个Tensor，包含所有输入Tensor非零数值的位置。

    参数：
        - **input** (Tensor) - Tensor的shape是 :math:`(x_1, x_2, ..., x_R)`。类型可以为int、float或bool。

    返回：
        一个2-D Tensor，数据类型为int64，包含所有输入中的非零数值的位置。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **ValueError** - 如果 `input` 的维度等于0。
