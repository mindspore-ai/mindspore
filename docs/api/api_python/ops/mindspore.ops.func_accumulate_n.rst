mindspore.ops.accumulate_n
==========================

.. py:function:: mindspore.ops.accumulate_n(x)

    逐元素计算列表中各个tensor的和。

    .. warning::
        该接口从2.8.0版本开始已被弃用，并将在2.9.0（不含）之后版本中被移除。建议使用 :func:`mindspore.ops.addn` 替代。

    :func:`mindspore.ops.accumulate_n` 与 :func:`mindspore.ops.addn` 类似，但accumulate_n不会等待其所有输入就绪后再求和，可节省内存。

    参数：
        - **x** (Union(tuple[Tensor], list[Tensor])) - 输入tensors。

    返回：
        Tensor
