mindspore.ops.logspace
======================

.. py:function:: mindspore.ops.logspace(start, end, steps, base=10.0, dtype=None)

    返回按照log scale平均分布的一组数值。

    在线性空间, 数值起始于:math:`base ** start`，结束于:math:`base ** end`。

    Args:
        - **start** (Union[int, list(int), tuple(int), Tensor]) - :math:`base ** start` 是点集的起始值。
        - **end** (Union[int, list(int), tuple(int), Tensor]) - :math:`base ** end` 是点集的结束值。
        - **steps** (int) - 构造Tensor的大小。
        - **base** (Union[int, float], 可选) - 对数函数的底。在 :math:`ln(samples) / ln(base)` (或 :math:`log_{base}(samples)`)之间的步长是一致的。默认值是10.0。
        - **dtype** (Union[:class:`mindspore.dtype`, str], 可选) - 执行计算的数据类型。如果 `dtype` 是None，从入参中推断数据类型。默认值是None。

    返回:
        按照log scale平均分布的一组Tensor数值。

    异常:
        - **TypeError** - 若参数的数据类型与上述不一致。
