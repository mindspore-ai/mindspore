mindspore.mint.nn.LogSoftmax
============================

.. py:class:: mindspore.mint.nn.LogSoftmax(dim=None)

    在指定轴上对输入Tensor应用LogSoftmax函数。对于指定轴上的切片 :math:`x`，其中每个元素为 :math:`x_i`，LogSoftmax函数的定义如下：

    .. math::
        \text{output}(x_i) = \log \left(\frac{\exp(x_i)} {\sum_{j = 0}^{N-1}\exp(x_j)}\right),

    其中， :math:`N` 为输入tensor在指定轴上的长度。

    参数：
        - **dim** (int, 可选) - 指定进行LogSoftmax的计算维度。默认 ``None`` 。

    返回：
        Tensor。
