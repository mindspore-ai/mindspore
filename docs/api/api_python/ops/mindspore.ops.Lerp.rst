mindspore.ops.Lerp
===================

.. py:class:: mindspore.ops.Lerp

    基于权重参数计算两个Tensor之间的线性插值。

    更多参考详见 :func:`mindspore.ops.lerp`。

    输入：
        - **start** (Tensor) - 进行线性插值的Tensor开始点，其数据类型必须为float16或者float32。
        - **end** (Tensor) - 进行线性插值的Tensor结束点，其数据类型必须与 `start` 一致。
        - **weight** (Union[float, Tensor]) - 线性插值公式的权重参数。为Scalar时，其数据类型为float。为Tensor时，其数据类型为float16或者float32。

    输出：
        Tensor，其数据类型和维度必须和输入中的 `start` 保持一致。
