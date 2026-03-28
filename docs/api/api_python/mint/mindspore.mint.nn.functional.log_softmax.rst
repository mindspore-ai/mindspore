mindspore.mint.nn.functional.log_softmax
========================================

.. py:function:: mindspore.mint.nn.functional.log_softmax(input, dim=None, *, dtype=None)

    在指定轴上对输入Tensor应用LogSoftmax函数。假设在指定轴上， :math:`x` 对应每个元素 :math:`x_i` ，则LogSoftmax函数如下所示：

    .. math::
        \text{output}(x_i) = \log \left(\frac{\exp(x_i)}{\sum_{j = 0}^{N-1}\exp(x_j)}\right)

    其中， :math:`N` 为Tensor长度。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **dim** (int, 可选) - 指定进行Log softmax运算的轴。默认值： ``None`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 输出数据类型。如果不为 ``None``，则输入会转化为 `dtype`。这有利于防止数值溢出。
          如果为 ``None``，则输出和输入的数据类型一致。默认值： ``None`` 。

    返回：
        Tensor，和输入Tensor的shape相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `dim` 不是int。
        - **ValueError** - 如果 `dim` 超出范围[-len(input.shape), len(input.shape))。
