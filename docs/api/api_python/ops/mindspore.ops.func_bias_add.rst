mindspore.ops.bias_add
===========================

.. py:function:: mindspore.ops.bias_add(input_x, bias)

    返回输入Tensor `input_x` 与偏置Tensor `bias` 之和。相加前会把偏置Tensor广播成与输入Tensor的shape一致。

    参数：
        - **input_x** (Tensor) - 输入Tensor。shape可以有2~5个维度。支持数据类型：

          - Ascend/CPU： all Number type。
          - GPU： float16、float32、int8。

        - **bias** (Tensor) - 偏置Tensor，shape为 :math:`(C)`。C必须与 `input_x` 的通道维度C相同。其数据类型与 `input_x` 一致。

    返回：
        Tensor，shape和数据类型与 `input_x` 相同。

    异常：
        - **TypeError** - `input_x` 或  `bias` 不是Tensor。
        - **TypeError** - `input_x` 与  `bias` 的数据类型不一致。
        - **TypeError** - `input_x` 的维度不在[2, 5]范围内。
