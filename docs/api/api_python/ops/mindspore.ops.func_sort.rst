mindspore.ops.sort
==================

.. py:function:: mindspore.ops.sort(input_x, axis=-1, descending=False)

    对输入tensor指定轴的元素进行排序。

    .. warning::
        - 2.9.0（不含）之后版本非兼容性变更：`axis` 将改为 `dim`，并新增 `stable` 参数。
        - 目前能良好支持的数据类型有：float16、uint8、int8、int16、int32、int64。如果使用float32，可能产生精度误差。

    .. note::
        当前Ascend后端只支持对最后一维进行排序。

    参数：
        - **input_x** (Tensor) - 输入tensor。
        - **axis** (int，可选) - 指定轴，默认 ``-1`` ，表示指定最后一维。
        - **descending** (bool，可选) - 排序方式。 ``True`` 表示按降序排列，否则按升序排列，默认 ``False`` 。

    返回：
        一个由tensor组成的tuple(sorted_tensor, indices)
