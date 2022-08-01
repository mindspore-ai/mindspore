mindspore.Tensor.fill
=====================

.. py:method:: mindspore.Tensor.fill(value)

    用标量值填充数组。

    .. note::
        与NumPy不同，Tensor.fill()将始终返回一个新的Tensor，而不是填充原来的Tensor。

    参数：
        - **value** (Union[None, int, float, bool]) - 所有元素都被赋予这个值。

    返回：
        Tensor，与原来的dtype和shape相同的Tensor。

    异常：
        - **TypeError** - 输入参数具有前面未指定的类型。