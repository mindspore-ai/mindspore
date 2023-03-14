mindspore.nn.LogSoftmax
=============================

.. py:class:: mindspore.nn.LogSoftmax(axis=-1)

    按元素计算Log Softmax激活函数。

    输入经Softmax函数、Log函数转换后，值的范围在[-inf,0)。

    Log Softmax定义如下：

    .. math::

        \text{logsoftmax}(x_i) = \log \left(\frac{\exp(x_i)}{\sum_{j=0}^{n-1} \exp(x_j)}\right),

    其中，:math:`x_{i}` 是输入Tensor的一个元素。

    参数：
        - **axis** (int) - Log Softmax运算的axis，-1表示最后一个维度。默认值：-1。

    输入：
        - **x** (Tensor) - Log Softmax的输入，数据类型为float16或float32。

    输出：
        Tensor，数据类型和shape与 `x` 相同，输出值的范围在[-inf,0)。

    异常：
        - **TypeError** - `axis` 不是int。
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。
        - **ValueError** - `axis` 不在[-len(x), len(x))范围中。
