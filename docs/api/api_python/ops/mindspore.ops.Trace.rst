mindspore.ops.Trace
====================

.. py:class:: mindspore.ops.Trace

    计算Tensor在对角线方向上元素的总和。

    .. note::
        输入必须是Tensor，复数类型暂不支持。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **x** (Tensor) - 二维Tensor。

    输出：
        Tensor，含有一个元素的零维Tensor，其数据类型与 `x` 一致。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
        - **ValueError** - 如果当 `x` 的维度不是2。
