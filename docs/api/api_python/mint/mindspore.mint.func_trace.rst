mindspore.mint.trace
====================

.. py:function:: mindspore.mint.trace(input)

    返回 `input` 的主对角线方向上的总和。

    参数：
        - **input** (Tensor) - 二维输入tensor。

    返回：
        Tensor，当 `input` 为数据类型为整型或bool时其数据类型为mindspore.int64，反之与 `input` 一致，含有一个元素。

    异常：
        - **ValueError** - `input` 的维度不是2。