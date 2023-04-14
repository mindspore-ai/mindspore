mindspore.ops.Polygamma
=======================

.. py:class:: mindspore.ops.Polygamma

    计算关于 `x` 的多伽马函数的 :math:`a` 阶导数。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.polygamma`。

    输入：
        - **a** (Tensor) - 多伽马函数求导的阶数， `a` 的shape为 :math:`()` ，支持的数据类型为int32和int64。
        - **x** (Tensor) - 用于计算多伽马函数的 :math:`a` 阶导数的Tensor，支持的数据类型为：float16、float32和float64。

    输出：
        Tensor。数据类型与 `x` 一致。
