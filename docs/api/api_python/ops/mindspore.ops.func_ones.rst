mindspore.ops.ones
===================

.. py:function:: mindspore.ops.ones(shape, dtype=None)

    创建一个值全为1的Tensor。

    第一个参数指定Tensor的shape，第二个参数指定填充值的数据类型。

    .. warning::
        参数 `shape` 在后续版本中将不再支持Tensor类型的输入。

    参数：
        - **shape** (Union[tuple[int], list[int], int, Tensor]) - 指定输出Tensor的shape，只允许正整数或者包含正整数的tuple/list/Tensor。
          如果是一个Tensor，必须是一个数据类型为int32或者int64的0-D或1-D Tensor。
        - **dtype** (:class:`mindspore.dtype`) - 用来描述所创建的Tensor的dtype。如果为 ``None`` ，那么将会使用mindspore.float32。默认值： ``None`` 。

    返回：
        Tensor，shape和dtype由输入定义。

    异常：
        - **TypeError** - 如果 `shape` 不是一个int，或元素为int的元组/列表/Tensor。。
