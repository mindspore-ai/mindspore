mindspore.mint.lerp
===================

.. py:function:: mindspore.mint.lerp(input, end, weight) -> Tensor

    基于权重参数计算两个tensor之间的线性插值。

    如果权重参数 `weight` 是一个Tensor，则 `input` 与 `end` 与 `weight` 广播后应有相同shape 。
    如果权重参数 `weight` 是一个浮点数，则 `input` 与 `end` 广播后shape应相同。
    如果权重参数 `weight` 是一个浮点数并且平台为Ascend, 则 `input` 与 `end` 应为float32。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. math::
        output_{i} = input_{i} + weight_{i} * (end_{i} - input_{i})

    参数：
        - **input** (Tensor) - 进行线性插值的tensor开始点，其数据类型必须为float16或者float32。
        - **end** (Tensor) - 进行线性插值的tensor结束点，其数据类型必须与 `input` 一致。
        - **weight** (Union[float, Tensor]) - 线性插值公式的权重参数。为Scalar时，其数据类型为float；为Tensor时，其数据类型为float16或者float32。

    返回：
        Tensor，其数据类型和维度必须和输入中的 `input` 保持一致。

    异常：
        - **TypeError** - 如果 `input` 或者 `end` 不是Tensor。
        - **TypeError** - 如果 `weight` 不是float类型Scalar或者Tensor。
        - **TypeError** - 如果 `input` 或者 `end` 的数据类型不是float16或者float32。
        - **TypeError** - 如果 `weight` 为Tensor且 `weight` 不是float16或者float32。
        - **TypeError** - 如果 `input` 和 `end` 的数据类型不一致。
        - **TypeError** - 如果 `weight` 为Tensor且 `input` 、 `end` 和 `weight` 数据类型不一致。
        - **ValueError** - 如果 `input` 与 `end` 的shape无法广播至一致。
        - **ValueError** - 如果 `weight` 为Tensor且 `weight` 与 `input` 的shape无法广播至一致。
