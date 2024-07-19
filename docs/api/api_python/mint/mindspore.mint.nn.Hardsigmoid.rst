mindspore.mint.nn.Hardsigmoid
==============================

.. py:class:: mindspore.mint.nn.Hardsigmoid

    逐元素计算Hard Sigmoid激活函数。

    Hard Sigmoid定义为：

    .. math::
        \text{Hardsigmoid}(input) =
        \begin{cases}
        0, & \text{ if } input ≤ -3, \\
        1, & \text{ if } input ≥ +3, \\
        input/6 + 1/2, & \text{ otherwise }
        \end{cases}

    HSigmoid函数图：

    .. image:: ../images/HSigmoid.png
        :align: center

    输入：
        - **input** (Tensor) - Hard Sigmoid的输入。支持数据类型：

          - Ascend：int32、float16、float32、bfloat16。
          - CPU/GPU：int8、int16、int32、int64、float16、float32、float64。

    输出：
        Tensor，数据类型和shape与 `input` 的相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 不是int或者float类型。
