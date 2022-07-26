mindspore.ops.ones
===================

.. py:function:: mindspore.ops.ones(shape, type)

    创建一个值全为1的Tensor。

    第一个参数指定Tensor的shape，第二个参数指定填充值的数据类型。

    参数：
        - **shape** (Union[tuple[int], int]) - 指定输出Tensor的shape，只能是正整数常量。
        - **type** (mindspore.dtype) - 指定输出Tensor的数据类型，只能是常量值。

    返回：
        Tensor，shape和数据类型与输入相同。

    异常：
        - **TypeError** - `shape` 既不是tuple，也不是int。