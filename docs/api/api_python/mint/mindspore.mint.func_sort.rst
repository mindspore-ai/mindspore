mindspore.mint.sort
====================

.. py:function:: mindspore.mint.sort(input, *, dim=-1, descending=False, stable=False)

    按指定顺序对输入Tensor的指定维上的元素进行排序。

    .. warning::
        目前能良好支持的数据类型有：float16、uint8、int8、int16、int32、int64。如果使用float32，可能产生精度误差。

    参数：
        - **input** (Tensor) - 进行排序的Tensor，shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的的额外维度。

    关键字参数：
        - **dim** (int，可选) - 指定排序的轴。默认值： ``-1`` ，表示指定最后一维。
        - **descending** (bool，可选) - 按降序还是升序。如果为 ``True`` ，则元素按降序排列，否则按升序排列。默认值： ``False`` 。
        - **stable** (bool，可选) - 按稳定排序还是非稳定排序。如果为 ``True`` ，则是稳定排序，否则为非稳定排序。默认值： ``False`` 。

    返回：
        - **y1** (Tensor) - 排序后的值，其shape和数据类型与输入一致。
        - **y2** (Tensor) - 返回值在原输入Tensor里对应的索引，数据类型为int64。

    异常：
        - **TypeError** - `dim` 不是int类型。
        - **TypeError** -  `descending` 不是bool类型。
        - **TypeError** - `input` 不是float16、float32、uint8、int8、int16、int32、int64或bfloat16。
        - **TypeError** -  `stable` 不是bool类型。
        - **ValueError** - `dim` 不在[-len(input.shape), len(input.shape))范围内。
