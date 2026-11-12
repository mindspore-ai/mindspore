mindspore.ops.repeat_elements
===============================

.. py:function:: mindspore.ops.repeat_elements(x, rep, axis=0)

    在指定轴上复制输入tensor的元素，类似 ``numpy.repeat`` 的功能。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        推荐使用 :func:`mindspore.mint.repeat_interleave` ，输入 `x` 的维度最大可支持8，并获得更好的性能。

    参数：
        - **x** (Tensor) - 输入tensor。
        - **rep** (int) - 复制次数，为正数。
        - **axis** (int) - 指定轴，默认 ``0`` 。

    返回：
        Tensor
