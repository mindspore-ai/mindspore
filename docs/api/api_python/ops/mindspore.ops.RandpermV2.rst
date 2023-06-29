mindspore.ops.RandpermV2
========================

.. py:class:: mindspore.ops.RandpermV2(dtype=mstype.int64)

    生成从0到n-1不重复的n个随机整数。

    更多参考详见 :func:`mindspore.ops.randperm`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **dtype** (mindspore.dtype, 可选) - 输出的类型。必须是以下类型之一：int32、int16、int8、uint8、int64、float64、float32、float16。默认值： ``mstype.int64`` 。


    输入：
        - **n** (Union[Tensor, int]) - 输入大小，如果为Tensor，则shape为 :math:`()` 或 :math:`(1,)` ，数据类型为int64。数值必须大于0。
        - **seed** (int, 可选) - 随机种子。默认值： ``0`` 。当 `seed` 为 ``-1`` （只有负值）时， `offset` 为 ``0`` ，随机数由时间决定。
        - **offset** (int, 可选) - 偏移量，生成随机数，优先级高于随机种子。 默认值： ``0`` 。必须是非负数。

    输出：
        Tensor，shape由参数 `n` 决定，dtype由参数 `dtype` 决定。
