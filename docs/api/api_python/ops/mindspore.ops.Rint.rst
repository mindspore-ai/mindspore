mindspore.ops.Rint
==================

.. py:class:: mindspore.ops.Rint

    逐元素计算最接近输入数据的整数。

    输入：
        - **input_x** (Tensor) - 任意维度输入Tensor，数据必须是float16、float32或者float64。

    输出：
        Tensor，shape和数据类型与 `input_x` 相同。

    异常：
        - **TypeError** - 如果 `input_x` 的数据类型不是float16、float32、float64。
