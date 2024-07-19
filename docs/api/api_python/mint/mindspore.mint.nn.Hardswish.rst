mindspore.mint.nn.Hardswish
============================

.. py:class:: mindspore.mint.nn.Hardswish

    逐元素计算Hard Swish。

    Hard Swish定义如下：

    .. math::
        \text{Hardswish}(input) =
        \begin{cases}
        0, & \text{ if } input ≤ -3, \\
        input, & \text{ if } input ≥ +3, \\
        input·(input + 3)/6, & \text{ otherwise }
        \end{cases}

    HSwish函数图：

    .. image:: ../images/HSwish.png
        :align: center

    输入：
        - **input** (Tensor) - Hard Swish的输入。支持数据类型：

          - Ascend：float16、float32、bfloat16。
          - CPU/GPU：int8、int16、int32、int64、float16、float32、float64。

    输出：
        Tensor，具有与 `input` 相同的数据类型和shape。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 不是int或者float类型。
