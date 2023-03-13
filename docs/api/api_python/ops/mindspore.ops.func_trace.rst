mindspore.ops.trace
===================

.. py:function:: mindspore.ops.trace(input)

    返回input的对角线方向上的总和。

    .. note::
        输入必须是Tensor，复数类型暂不支持。

    输入：
        - **input** (Tensor) - 二维Tensor。

    输出：
        Tensor，其数据类型与 `input` 一致，含有一个元素。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **ValueError** - 如果当 `input` 的维度不是2。