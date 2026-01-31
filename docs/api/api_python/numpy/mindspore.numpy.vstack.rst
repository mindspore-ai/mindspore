mindspore.numpy.vstack
=================================

.. py:function:: mindspore.numpy.vstack(tup)

    按顺序垂直堆叠Tensor。此操作等同于沿第一轴进行连接。对于一维Tensor，应首先将其shape变为 ``(1, N)`` ，然后沿第一轴进行连接。

    参数：
        - **tup** (Union[Tensor, tuple, list]) - 一组一维或二维Tensor。Tensor必须在除第一轴之外的所有轴上具有相同的shape。一维Tensor必须具有相同的shape。

    返回：
        堆叠后的Tensor，由给定的Tensor堆叠而成。

    异常：
        - **TypeError** - 如果 `tup` 不是Tensor、list或tuple。
        - **ValueError** - 如果 `tup` 为空。