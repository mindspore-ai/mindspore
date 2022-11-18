mindspore.ops.arange
=====================

.. py:function:: mindspore.ops.arange(start=0, end=None, step=1, *, dtype=None)

    返回从 `start` 开始，步长为 `step` ，且不超过 `end` （不包括 `end` ）的序列。

    参数：
        - **start** (Union[float, int, Tensor]) - 序列中的第一个数字。
        - **end** (Union[float, int, Tensor]) - 序列的上限或下限，不包含在序列中。
        - **step** (Union[float, int, Tensor]) - 表述序列中数值的步长。
        - **dtype** (mindspore.dtype, 可选) - 返回序列的数据类型。默认值：None。如果未指定或者为None，将会被推断为 `start` 、 `end` 和 `step` 参数中精度最高的类型。

    返回：
        一维Tensor，数据类型与输入数据类型一致。

    异常：
        - **TypeError** - `start` ， `end` ， `step` 既不是int也不是float也不是在支持类型中的TensorScalar。
        - **ValueError** - `step` 等于0。
        - **ValueError** - `start` 小于等于 `end` ， `step` 小于0。
        - **ValueError** - `start` 大于等于 `end` ， `step` 大于0。
