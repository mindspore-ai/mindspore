mindspore.ops.fliplr
=====================

.. py:function:: mindspore.ops.fliplr(x)

    沿左右方向翻转Tensor中每行的元素。
    Tensor的列会被保留，但显示顺序将与以前不同。

    参数：
        - **x** (Tensor) - 输入tensor。

    返回：
        Tensor。

    异常：
        - **TypeError** - `x` 不是Tensor。
