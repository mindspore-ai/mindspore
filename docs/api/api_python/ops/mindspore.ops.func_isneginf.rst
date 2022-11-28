mindspore.ops.isneginf
======================

.. py:function:: mindspore.ops.isneginf(x)

    逐元素判断是否是负inf。

    参数：
        - **x** (Tensor) - 输入Tensor。

    返回：
        Tensor，对应 `x` 元素为负inf的位置是true，反之为false。

    异常：
        - **TypeError** - `x` 不是Tensor。
