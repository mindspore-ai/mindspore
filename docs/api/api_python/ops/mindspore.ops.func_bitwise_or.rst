mindspore.ops.bitwise_or
========================

.. py:function:: mindspore.ops.bitwise_or(input, other)

    逐元素执行两个Tensor的或运算。

    .. math::

        out_i = input_{i} \mid other_{i}

    输入 `input` 和 `other` 遵循 `隐式类型转换规则 <https://www.mindspore.cn/docs/zh-CN/master/note/operator_list_implicit.html>`_ ，使数据类型保持一致。
    如果 `input` 和 `other` 数据类型不同，低精度数据类型将自动转换成高精度数据类型。

    参数：
        - **input** (Tensor) - 第一个输入Tensor，其shape为 :math:`(N, *)` ，其中 :math:`*` 为任意数量的额外维度。
        - **other** (Tensor) - 第二个输入Tensor，数据类型与 `input` 一致。

    返回：
        Tensor，是一个与 `input` 相同类型的Tensor。

    异常：
        - **TypeError** - `input` 或 `other` 不是Tensor。
