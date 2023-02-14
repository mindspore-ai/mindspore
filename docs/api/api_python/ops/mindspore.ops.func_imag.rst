mindspore.ops.imag
===================

.. py:function:: mindspore.ops.imag(input)

    返回包含输入Tensor的虚部。如果输入为实数，则返回零。

    参数：
        - **input** (Tensor) - 要计算的输入Tensor。

    返回：
        Tensor，shape与 `input` 相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
