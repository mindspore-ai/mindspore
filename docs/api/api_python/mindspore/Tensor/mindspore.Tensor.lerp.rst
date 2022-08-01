mindspore.Tensor.lerp
=====================

.. py:method:: mindspore.Tensor.lerp(end, weight)

    基于某个浮点数Scalar或权重Tensor的值， 计算当前Tensor和 `end` Tensor之间的线性插值。

    如果参数 `weight` 是一个Tensor，那么另两个输入的维度信息可以被广播到当前Tensor。
    如果参数 `weight` 是一个Scalar， 那么 `end` 的维度信息可以被广播到当前Tensor。

    参数：
        - **end** (Tensor) - 进行线性插值的Tensor结束点，其数据类型必须为float16或者float32。
        - **weight** (Union[float, Tensor]) - 线性插值公式的权重参数。当为Scalar时，其数据类型为float，当为Tensor时，其数据类型为float16或者float32。

    返回：
        返回新的Tensor，其数据类型和维度必须和输入中的当前Tensor保持一致。

    异常：
        - **TypeError** - 如果 `end` 不是Tensor。
        - **TypeError** - 如果 `weight` 不是float类型Scalar或者Tensor。
        - **TypeError** - 如果 `end` 的数据类型不是float16或者float32。
        - **TypeError** - 如果 `weight` 为Tensor且 `weight` 不是float16或者float32。
        - **TypeError** - 如果当前Tensor和 `end` 的数据类型不一致。
        - **TypeError** - 如果 `weight` 为Tensor且 `end` 、 `weight` 和当前Tensor数据类型不一致。
        - **ValueError** - 如果 `end` 的维度信息无法相互广播到当前Tensor。
        - **ValueError** - 如果 `weight` 为Tensor且 `weight` 的维度信息无法广播到当前Tensor。