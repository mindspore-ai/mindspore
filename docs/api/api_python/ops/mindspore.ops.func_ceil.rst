mindspore.ops.ceil
===================

.. py:function:: mindspore.ops.ceil(input)

    向上取整函数。

    .. math::
        out_i = \lceil x_i \rceil = \lfloor x_i \rfloor + 1

    参数：
        - **input** (Tensor) - Ceil的输入。支持的数据类型:

          - Ascend: float16、float32、float64、bfloat16。
          - GPU/CPU: float16、float32、float64。

    返回：
        Tensor，shape与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - 如果 `input` 的数据类型不是float16、float32、float64或bfloat16。
