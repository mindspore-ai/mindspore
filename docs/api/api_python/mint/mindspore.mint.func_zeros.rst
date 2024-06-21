mindspore.mint.zeros
=====================

.. py:function:: mindspore.mint.zeros(size, *, dtype=None)

    创建一个值全为0的Tensor。第一个参数 `size` 指定Tensor的shape，第二个参数 `dtype` 指定填充值的数据类型。

    参数：
        - **size** (Union[tuple[int], list[int], int, Tensor]) - 用来描述所创建的Tensor的shape，只允许正整数或者包含正整数的tuple/list/Tensor。
          如果是一个Tensor，必须是一个数据类型为int32或者int64的0-D或1-D Tensor。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 用来描述所创建的Tensor的dtype。如果为 ``None`` ，那么将会使用mindspore.float32。默认值： ``None`` 。

    返回：
        Tensor，dtype和shape由入参决定。

    异常：
        - **TypeError** - 如果 `size` 不是一个int，或元素为int的元组/列表/Tensor。
