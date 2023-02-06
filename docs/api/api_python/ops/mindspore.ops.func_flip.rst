mindspore.ops.flip
===================

.. py:function:: mindspore.ops.flip(x, dims)

    沿给定轴翻转Tensor中元素的顺序。

    Tensor的shape会被保留，但是元素将重新排序。

    参数：
        - **x** (Tensor) - 输入Tensor。
        - **dims** (Union[list[int], tuple[int]]) - 需要翻转的一个轴或多个轴。在元组中指定的所有轴上执行翻转，如果 `dims` 是一个包含负数的整数元组，则该轴为按倒序计数的轴位置。

    返回：
        返回沿给定轴翻转计算结果的Tensor。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **ValueError** - `dims` 为None。
        - **ValueError** - `dims` 不为int组成的tuple。
