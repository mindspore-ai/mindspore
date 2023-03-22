mindspore.ops.Zeros
====================

.. py:class:: mindspore.ops.Zeros

    创建一个值全为0的Tensor。

    第一个参数指定Tensor的shape，第二个参数指定填充值的数据类型。

    输入：
        - **shape** (Union[tuple[int], int]) - 指定输出Tensor的shape。
        - **type** (mindspore.dtype) - 指定输出Tensor的数据类型。
    输出：
        Tensor，数据类型和shape与输入shape相同。

    异常：
        - **TypeError** - `shape` 既不是int也不是tuple。
        - **TypeError** - `shape` 是tuple，其元素并非全部是int。