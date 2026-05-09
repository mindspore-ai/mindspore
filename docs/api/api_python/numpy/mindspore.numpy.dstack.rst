mindspore.numpy.dstack
=================================

.. py:function:: mindspore.numpy.dstack(tup)

    按顺序在深度方向（沿第三轴）堆叠Tensor。此操作等同于沿第三轴进行连接。对于一维Tensor :math:`(N,)` ，应将其shape变为 :math:`(1, N, 1)` 。对于二维Tensor :math:`(M, N)` ，应将其shape变为 :math:`(M, N, 1)` 后再进行连接。

    参数：
        - **tup** (Union[Tensor, tuple, list]) - 一组Tensor。Tensor必须在除第三轴之外的所有轴上具有相同的shape。一维或二维Tensor必须具有相同的shape。

    返回：
        堆叠后的Tensor，由给定的Tensor堆叠而成。

    异常：
        - **TypeError** - 如果 ``tup`` 不是Tensor、list或tuple。
        - **ValueError** - 如果 ``tup`` 为空。