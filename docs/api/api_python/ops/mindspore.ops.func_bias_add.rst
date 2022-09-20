mindspore.ops.bias_add
======================

.. py:function:: mindspore.ops.bias_add(input_x, bias)

    返回输入Tensor与偏置Tensor之和。相加前会把偏置Tensor广播成与输入Tensor的shape一致。

    参数：
        - **input_x** (Tensor) - 输入Tensor。shape可以有2~5个维度。数据类型应为float16或float32。
        - **bias** (Tensor) - 偏置Tensor，shape为 :math:`(C)`。C必须与 `input_x` 的通道维度C相同，数据类型应为float16或float32。

    返回：
        Tensor，shape和数据类型与 `input_x` 相同。

    异常：
        - **TypeError** - `input_x` 或 `bias` 不是Tensor。
        - **TypeError** - `input_x` 或 `bias` 的数据类型既不是float16也不是float32。
        - **TypeError** - `input_x` 或 `bias` 的数据类型不一致。
        - **TypeError** - `input_x` 的维度不在[2, 5]范围内。
