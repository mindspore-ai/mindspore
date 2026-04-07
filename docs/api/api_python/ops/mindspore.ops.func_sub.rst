mindspore.ops.sub
=================

.. py:function:: mindspore.ops.sub(input, other)

    逐元素计算第一个输入减去第二个输入的值。

    .. math::

        out_{i} = input_{i} - other_{i}

    .. note::
        - 当两个输入具有不同的shape时，它们的shape必须要能广播为一个共同的shape。
        - 两个输入不能同时为bool类型。[True, Tensor(True), Tensor(np.array([True]))]等都为bool类型。
        - 支持隐式类型转换、类型提升。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将新增可选关键字参数 `alpha` 。
        接口签名将变更为 ``mindspore.ops.sub(input, other, *, alpha=1)`` 。
        参数 `alpha` 用于对 `other` 进行标量缩放，公式将变更为 :math:`out_{i} = input_{i} - alpha \times other_{i}` 。
        该变更保持向后兼容；未显式指定 `alpha` 时，行为保持不变。

    参数：
        - **input** (Union[Tensor, number.Number, bool]) - 第一个输入。
        - **other** (Union[Tensor, number.Number, bool]) - 第二个输入。

    返回：
        Tensor
