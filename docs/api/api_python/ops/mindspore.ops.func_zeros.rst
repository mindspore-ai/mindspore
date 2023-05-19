mindspore.ops.zeros
====================

.. py:function:: mindspore.ops.zeros(size, dtype=None)

    创建一个填满0的Tensor，shape由 `size` 决定， dtype由 `dtype` 决定。

    参数：
        - **size** (Union[tuple[int], int, Tensor]) - 用来描述所创建的Tensor的shape，只允许正整数或者包含正整数的tuple/Tensor。
          如果是一个Tensor，必须是一个数据类型为int32或者int64的0-D或1-D Tensor。
        - **dtype** (:class:`mindspore.dtype`, 可选) - 用来描述所创建的Tensor的dtype。如果为 ``None`` ，那么将会使用mindspore.float32。默认值： ``None`` 。

    返回：
        Tensor，dtype和shape由入参决定。

    异常：
        - **TypeError** - 如果 `size` 不是tuple、int或者Tensor。
