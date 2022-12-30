mindspore.ops.fills
===================

.. py:function:: mindspore.ops.fills(x, value)

    创建一个与输入Tensor具有相同shape和dtype的Tensor，并用指定值填充。

    参数：
        - **x** (Tensor) - 输入Tensor，用来指定输出Tensor的shape和dtype。数据类型为int8，int16，int32，float16，float32。
        - **value** (Union[int, float, Tensor]) - 用来填充输出Tensor的值。数据类型为int，float或零维Tensor。

    返回：
        Tensor，与输入数据 `x` 具有相同的shape和dtype。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `value` 具有前面未指定的类型。
        - **RuntimeError** - `value` 不能转换为与当前Tensor相同的类型。
        - **ValueError** - `value` 是非零维Tensor。
