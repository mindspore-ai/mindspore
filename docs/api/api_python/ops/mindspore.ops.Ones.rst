mindspore.ops.Ones
===================

.. py:class:: mindspore.ops.Ones

    创建一个值全为1的Tensor。

    更多参考详见 :func:`mindspore.ops.ones`。

    .. warning::
        参数 `shape` 在后续版本中将不再支持Tensor类型的输入。

    输入：
        - **shape** (Union[tuple[int], list[int], int, Tensor]) - 指定输出Tensor的shape。
        - **type** (:class:`mindspore.dtype`) - 指定输出Tensor的数据类型。

    输出：
        Tensor，shape和dtype由输入定义

    异常：
        - **TypeError** - 如果 `shape` 不是一个int，或元素为int的元组/列表/Tensor。
