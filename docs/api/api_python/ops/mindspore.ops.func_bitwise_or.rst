mindspore.ops.bitwise_or
========================

.. py:function:: mindspore.ops.bitwise_or(x, y)

    逐元素计算 :math:`x | y` 的值。

    .. math::

        out_i = x_{i} \mid y_{i}

    .. note::
        - 输入 `x` 和 `y` 遵循 `隐式类型转换规则 <https://www.mindspore.cn/docs/zh-CN/master/note/operator_list_implicit.html>`_ ，使数据类型保持一致。
        - 输入必须是两个Tensor。

    **参数：**

    - **x** (Tensor) - 第一个输入，是一个数据类型为uint8、uint16、unint32、uint64、int8、int16、int32或int64的Tensor。
    - **y** (Tensor) - 第二个输入，是一个与 `x` 相同类型的Tensor。

    **返回：**

    Tensor，是一个与 `x` 相同类型的Tensor。

    **异常：**

    - **TypeError** - `x` 或 `y` 不是Tensor。
    - **RuntimeError** - 输入的 `x` 与 `y` 不符合参数类型转换规则。
