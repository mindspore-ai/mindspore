mindspore.ops.deepcopy
======================

.. py:function:: mindspore.ops.deepcopy(input_x)

    返回输入Tensor的深拷贝。

    参数：
        - **input_x** (Tensor) - 输入Tensor，shape为 :math:`(x_1, x_2, ..., x_R)` 。

    返回：
        Tensor，`input_x` 的深拷贝。

    异常：
        - **TypeError** - 如果 `input_x` 不是Tensor。