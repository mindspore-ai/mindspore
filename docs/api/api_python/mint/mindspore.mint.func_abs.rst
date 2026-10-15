mindspore.mint.abs
==================

.. py:function:: mindspore.mint.abs(input)

    逐元素计算输入tensor的绝对值。

    对于复数输入，逐元素计算其幅值。

    .. math::
        out_i = |input_i|

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor，shape与 `input` 相同。当 `input` 的数据类型为 ``complex64`` 或 ``complex128`` 时，
        输出数据类型分别为 ``float32`` 或 ``float64`` ；其他情况下，输出数据类型与 `input` 相同。
