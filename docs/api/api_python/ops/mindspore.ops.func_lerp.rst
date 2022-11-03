mindspore.ops.lerp
==================

.. py:function:: mindspore.ops.lerp(start, end, weight)

    基于权重参数计算两个Tensor之间的线性插值。

    如果权重参数 `weight` 是一个Tensor，则各输入广播后应有相同shape 。
    如果权重参数 `weight` 是一个浮点数，则 `start` 与 `end` 广播后shape应相同。

    .. math::

            output_{i} = start_{i} + weight_{i} * (end_{i} - start_{i})

    参数：
        - **start** (Tensor) - 进行线性插值的Tensor开始点，其数据类型必须为float16或者float32。
        - **end** (Tensor) - 进行线性插值的Tensor结束点，其数据类型必须与 `start` 一致。
        - **weight** (Union[float, Tensor]) - 线性插值公式的权重参数。为Scalar时，其数据类型为float。为Tensor时，其数据类型为float16或者float32。

    返回：
        Tensor，其数据类型和维度必须和输入中的 `start` 保持一致。

    异常：
        - **TypeError** - 如果 `start` 或者 `end` 不是Tensor。
        - **TypeError** - 如果 `weight` 不是float类型Scalar或者Tensor。
        - **TypeError** - 如果 `start` 或者 `end` 的数据类型不是float16或者float32。
        - **TypeError** - 如果 `weight` 为Tensor且 `weight` 不是float16或者float32。
        - **TypeError** - 如果 `start` 和 `end` 的数据类型不一致。
        - **TypeError** - 如果 `weight` 为Tensor且 `start` 、 `end` 和 `weight` 数据类型不一致。
        - **ValueError** - 如果 `start` 与 `end` 的shape无法广播至一致。
        - **ValueError** - 如果 `weight` 为Tensor且 `weight` 与 `start` 的shape无法广播至一致。

