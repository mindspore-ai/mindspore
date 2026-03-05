mindspore.mint.sort
====================

.. py:function:: mindspore.mint.sort(input, *, dim=-1, descending=False, stable=False)

    对输入tensor指定维度的元素进行排序。

    .. warning::
        目前能良好支持的数据类型有：mindspore.float16、mindspore.uint8、mindspore.int8、mindspore.int16、mindspore.int32、mindspore.int64。如果使用mindspore.float32，可能产生精度误差。

    参数：
        - **input** (Tensor) - 输入tensor。

    关键字参数：
        - **dim** (int，可选) - 指定排序的维度。默认 ``-1`` ，表示指定最后一维。
        - **descending** (bool，可选) - 按降序还是升序排序。如果为 ``True`` ，则元素按降序排列，否则按升序排列，默认 ``False`` 。
        - **stable** (bool，可选) - 是否保证等效元素的顺序。默认 ``False`` 。

    返回：
        一个由tensor组成的tuple(sorted_tensor, indices)。

    异常：
        - **ValueError** - `dim` 不在[-len(input.shape), len(input.shape))范围内。
