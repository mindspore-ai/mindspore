mindspore.ops.isposinf
======================

.. py:function:: mindspore.ops.isposinf(input)

    逐元素判断是否是正inf。

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，对应 `input` 元素为正inf的位置是true，反之为false。

    异常：
        - **TypeError** - `input` 不是Tensor。
