mindspore.ops.log2
===================

.. py:function:: mindspore.ops.log2(input)

    逐元素返回Tensor以2为底的对数。

    .. math::
        y_i = log_{2}(input_i)

    .. warning::
        如果log2的输入值范围在(0, 0.01]或[0.95, 1.05]区间，输出精度可能会受影响。

    .. note::
        Ascend上输入Tensor的维度要小于等于8，CPU或GPU上输入Tensor的维度要小于8。

    参数：
        - **input** (Tensor) - 任意维度的输入Tensor。该值必须大于0。

    返回：
        Tensor，具有与 `input` 相同的shape。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - 在GPU和CPU平台上运行时，`input` 的数据类型不是float16、float32或float64。在Ascend平台上运行时，`input` 的数据类型不是float16或float32。
