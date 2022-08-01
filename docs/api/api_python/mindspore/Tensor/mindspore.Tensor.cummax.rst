mindspore.Tensor.cummax
=======================

.. py:method:: mindspore.Tensor.cummax(axis)

    返回一个元组（最值、索引），其中最值是输入张量 `x` 沿维度 `axis` 的累积最大值，索引是每个最大值的索引位置。

    .. math::
        \begin{array}{ll} \\
        y{i} = max(x{1}, x{2}, ... , x{i})
        \end{array}

    参数：
        - **axis** (int) - 算子操作的维度，维度的大小范围是[-x.ndim, x.ndim - 1]。

    返回：
        一个包含两个Tensor的元组，分别表示累积最大值和对应索引。

    异常：
        - **TypeError** - 如果 `axis` 不是int。
        - **ValueError** - 如果 `axis` 不在范围[-x.ndim, x.ndim - 1]内。