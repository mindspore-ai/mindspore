mindspore.ops.linspace
======================

.. py:function:: mindspore.ops.linspace(start, end, steps)

    创建一个steps个元素的，在[start, end]区间内均匀分布的一维tensor。

    .. math::
        \begin{aligned}
        &step = (end - start)/(steps - 1)\\
        &output = [start, start+step, start+2*step, ... , end]
        \end{aligned}

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将新增可选关键字参数 `dtype` ，接口签名将变更为
        ``mindspore.ops.linspace(start, end, steps, *, dtype=None)`` 。
        该变更与当前版本兼容，不使用 `dtype` 参数时行为不变。

    参数：
        - **start** (Union[Tensor, int, float]) - 区间的起始值。
        - **end** (Union[Tensor, int, float]) - 区间的末尾值。
        - **steps** (Union[Tensor, int]) - 元素数量。

    返回：
        Tensor
