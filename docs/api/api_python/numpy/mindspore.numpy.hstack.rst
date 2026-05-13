mindspore.numpy.hstack
=================================

.. py:function:: mindspore.numpy.hstack(tup)

    按顺序水平堆叠Tensor。此操作等效于沿第二轴连接Tensor，但当输入为一维Tensor时，则沿第一轴连接。

    参数：
        - **tup** (Union[Tensor, tuple, list]) - 一组一维或二维Tensor。Tensor在除第二轴外的所有轴上必须具有相同的shape，一维Tensor的长度可以不同。

    返回：
        堆叠后的Tensor，由给定的Tensor堆叠而成。

    异常：
        - **TypeError** - 如果 `tup` 不是Tensor、list或tuple。
        - **ValueError** - 如果 `tup` 为空。
