mindspore.ops.log_softmax
=========================

.. py:function:: mindspore.ops.log_softmax(logits, axis=-1)

    在指定轴上对输入Tensor应用LogSoftmax函数。假设在指定轴上， :math:`x` 对应每个元素 :math:`x_i` ，则LogSoftmax函数如下所示：

    .. math::
        \text{output}(x_i) = \log \left(\frac{\exp(x_i)} {\sum_{j = 0}^{N-1}\exp(x_j)}\right),

    其中， :math:`N` 为Tensor长度。

    参数：
        - **logits** (Tensor) - shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度，其数据类型为float16或float32。
        - **axis** (int) - 指定进行运算的轴。默认值： ``-1`` 。

    返回：
        Tensor，数据类型和shape与 `logits` 相同。

    异常：
        - **TypeError** - `axis` 不是int。
        - **TypeError** - `logits` 的数据类型既不是float16也不是float32。
        - **ValueError** - `axis` 不在[-len(logits.shape), len(logits.shape))范围中。
        - **ValueError** - `logits` 的维度小于1。
