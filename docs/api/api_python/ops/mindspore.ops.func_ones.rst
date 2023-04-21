mindspore.ops.ones
===================

.. py:function:: mindspore.ops.ones(shape, dtype=None)

    创建一个值全为1的Tensor。

    第一个参数指定Tensor的shape，第二个参数指定填充值的数据类型。

    参数：
        - **shape** (Union[tuple[int], int]) - 指定输出Tensor的shape。
        - **dtype** (:class:`mindspore.dtype`) - 用来描述所创建的Tensor的 `dtype`。如果为 ``None`` ，那么将会使用mindspore.float32。默认值： ``None`` 。

    返回：
        Tensor，shape和数据类型与输入相同。

    异常：
        - **TypeError** - `shape` 既不是tuple，也不是int。