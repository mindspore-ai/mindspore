mindspore.Tensor.fills
======================

.. py:method:: mindspore.Tensor.fills(value)

    创建一个与当前Tensor具有相同shape和type的Tensor，并用标量值填充。

    .. note::
        与NumPy不同，Tensor.fills()将始终返回一个新的Tensor，而不是填充原来的Tensor。

    参数：
        - **value** (Union[int, float, Tensor]) - 用来填充输出Tensor的值。数据类型为int，float或0-维Tensor。

    返回：
        Tensor，与当前Tensor具有相同的shape和type。

    异常：
        - **TypeError** - `value` 具有前面未指定的类型。
        - **RuntimeError** - `value` 不能转换为与当前Tensor相同的类型。
        - **ValueError** - `value` 是非0维Tensor。