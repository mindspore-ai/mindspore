mindspore.ops.Trace
====================

.. py:class:: mindspore.ops.Trace

    返回在Tensor的对角线方向上的总和。

    .. note::
        输入必须是Tensor，复数类型暂不支持。

    输入：
        - **x** (Tensor) - 二维Tensor。

    输出：
        Tensor，其数据类型与 `x` 一致，含有一个元素。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
        - **ValueError** - 如果当 `x` 的维度不是2。
