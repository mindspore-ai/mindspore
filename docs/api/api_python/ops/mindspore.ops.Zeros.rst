mindspore.ops.Zeros
====================

.. py:class:: mindspore.ops.Zeros

    创建一个值全为0的Tensor。

    第一个参数指定Tensor的shape，第二个参数指定填充值的数据类型。

    .. warning::
        参数 `shape` 在后续版本中将不再支持Tensor类型的输入。

    输入：
        - **shape** (Union[tuple[int], list[int], int, Tensor]) - 指定输出Tensor的shape。
        - **type** (mindspore.dtype) - 指定输出Tensor的数据类型。

    输出：
        Tensor，shape和dtype由输入定义。

    异常：
        - **TypeError** - 如果 `shape` 不是一个int，或元素为int的元组/列表/Tensor。
