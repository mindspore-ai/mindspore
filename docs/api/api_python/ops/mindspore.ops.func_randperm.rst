mindspore.ops.randperm
========================

.. py:function:: mindspore.ops.randperm(n, seed=0, offset=0, dtype=mstype.int64)

    生成从 0 到 n-1 的整数随机排列。

    返回由 n 推断出的具有确定形状的张量，其中的随机数取自给定类型可以表示的数据范围。

    .. note::
        `n` 必须大于0。

    返回一个Tensor，shape和dtype由输入决定，其元素为服从标准正态分布的 :math:`[0, 1)` 区间的数字。

    参数：
        - **n** (Union[Tensor, int]) - 输入大小，如果为Tensor，则形状为()或(1,)，数据类型为int64。
        - **seed** (int，可选): 随机种子。 默认值：0。当seed为-1（只有负值）时，offset为0，由时间决定。
        - **offset** (int，可选): 优先级高于随机种子。 默认值：0。必须是非负数。
        - **dtype** (:class:`mindspore.dtype`，可选)：输出的类型。必须是以下类型之一：int32、int16、int8、uint8、int64、float64、float32、float16。 默认值：int64。

    返回：
        Tensor，shape由参数 `n` 决定，dtype由参数 `dtype` 决定。

    异常：
        - **TypeError** - 如果 `dtype` 不是一个 `mstype.float_type` 类型。
        - **ValueError** - 如果 `n` 是负数或0。
        - **ValueError** - 如果 `seed` 不是非负整数。
        - **ValueError** - 如果 `n` 是超过指定数据类型的最大范围。
