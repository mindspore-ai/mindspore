mindspore.mint.linspace
=======================

.. py:function:: mindspore.mint.linspace(start, end, steps, *, dtype=None)

    创建一个steps个元素的，在[start, end]区间内均匀分布的一维tensor。

    .. math::
        \begin{aligned}
        &step = (end - start)/(steps - 1)\\
        &output = [start, start+step, start+2*step, ... , end]
        \end{aligned}

    参数：
        - **start** (Union[float, int]) - 区间的起始值。
        - **end** (Union[float, int]) - 区间的末尾值。
        - **steps** (int) - 元素数量。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 返回的数据类型。默认 ``None``。

    返回：
        Tensor
