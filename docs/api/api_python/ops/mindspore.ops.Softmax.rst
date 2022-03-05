mindspore.ops.Softmax
======================

.. py:class:: mindspore.ops.Softmax(axis=-1)

    Softmax函数。

    在指定轴上使用Softmax函数做归一化操作。假设指定轴 :math:`x` 上有切片，那么每个元素 :math:`x_i` 所对应的Softmax函数如下所示：

    .. math::
        \text{output}(x_i) = \frac{exp(x_i)}{\sum_{j = 0}^{N-1}\exp(x_j)},

    其中 :math:`N` 代表Tensor的长度。

    **参数：**

    - **axis** (Union[int, tuple]) - 指定Softmax操作的轴。默认值：-1。

    **输入：**

    - **logits** (Tensor) - Softmax的输入，任意维度的Tensor。其数据类型为float16或float32。

    **输出：**

    Tensor，数据类型和shape与 `logits` 相同。

    **异常：**

    - **TypeError** - `axis` 既不是int也不是tuple。
    - **TypeError** - `logits` 的数据类型既不是float16也不是float32。
    - **ValueError** - `axis` 是长度小于1的tuple。
    - **ValueError** - `axis` 是一个tuple，其元素不全在[-len(logits.shape), len(logits.shape))范围中。
    