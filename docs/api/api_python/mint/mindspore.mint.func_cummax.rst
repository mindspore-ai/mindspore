mindspore.mint.cummax
======================

.. py:function:: mindspore.mint.cummax(input, dim)

    返回一个元组（最值、索引），其中最值是输入Tensor `input` 沿维度 `dim` 的累积最大值，索引是每个最大值的索引位置。

    .. math::
        \begin{array}{ll} \\
            y_{i} = \max(x_{1}, x_{2}, ... , x_{i})
        \end{array}

    参数：
        - **input** (Tensor) - 输入Tensor，要求维度大于0。
        - **dim** (int) - 算子操作的维度，维度的大小范围是[-input.ndim, input.ndim - 1]。

    返回：
        一个包含两个Tensor的元组，分别表示累积最大值和对应索引。每个输出Tensor的形状和输入Tensor的形状相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `dim` 不是int。
        - **ValueError** - 如果 `dim` 不在范围[-input.ndim, input.ndim - 1]内。

    .. note::
        Ascend不支持O2模式。
