mindspore.ops.logspace
======================

.. py:function:: mindspore.ops.logspace(start, end, steps, base=10, *, dtype=mstype.float32)

    返回一个 `steps` 个元素的，在 [ :math:`base^{start}` , :math:`base^{end}` ] 区间内均匀分布的一维tensor。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：关键字参数 `dtype` 的默认值将从 ``mstype.float32`` 变更为 ``None`` 。

        变更后的接口签名为 ``mindspore.ops.logspace(start, end, steps, base=10, *, dtype=None)`` 。

    .. math::
        \begin{aligned}
        &step = (end - start)/(steps - 1)\\
        &output = [base^{start}, base^{start + 1 * step}, ... , base^{start + (steps-2) * step}, base^{end}]
        \end{aligned}

    参数：
        - **start** (Union[float, Tensor]) - 区间的起始值。
        - **end** (Union[float, Tensor]) - 区间的末尾值。
        - **steps** (int) - 元素数量。
        - **base** (int，可选) - 对数函数的底数。默认 ``10`` 。

    关键字参数：
        - **dtype** (mindspore.dtype，可选) - 指定数据类型。默认 ``mstype.float32`` 。2.9.0（不含）之后版本默认值将变更为 ``None`` 。

    返回：
        Tensor
