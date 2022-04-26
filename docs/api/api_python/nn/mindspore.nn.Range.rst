mindspore.nn.Range
===================

.. py:class:: mindspore.nn.Range(start, limit=None, delta=1)

    根据指定步长在范围[start, limit)中创建数字序列。

    输出的数据长度为 :math:`\left \lfloor \frac{limit-start}{delta}  \right \rfloor + 1` 并且 `delta` 是Tensor中两个值之间的间隔。

    .. math::
        out_{i+1} = out_{i} + delta

    **参数：**

    - **start** (Union[int, float]) - 如果 `limit` 为None，则该值在范围内充当结束，0为起始。否则， `start` 将充当范围中的起始。
    - **limit** (Union[int, float]) - 序列的上限。如果为None，则默认为 `start` 的值，同时将范围内的0作为起始。它不能等于 `start` 。默认值：None。
    - **delta** (Union[int, float]) - 指定步长。不能等于零。默认值：1。

    **输出：**

    Tensor，如果 `start` 、 `limit` 和 `delta` 的数据类型都是int，则数据类型为int。否则，数据类型为float。