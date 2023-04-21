mindspore.ops.repeat_elements
===============================

.. py:function:: mindspore.ops.repeat_elements(x, rep, axis=0)

    在指定轴上复制输入Tensor的元素，类似 `np.repeat` 的功能。

    参数：
        - **x** (Tensor) - 输入Tensor。类型为float16、float32、int8、uint8、int16、int32或int64。
        - **rep** (int) - 指定复制次数，为正数。
        - **axis** (int) - 指定复制轴，默认值： ``0`` 。

    返回：
        Tensor，值沿指定轴复制。如果 `x` 的shape为 :math:`(s1, s2, ..., sn)` ，轴为i，则输出的shape为 :math:`(s1, s2, ..., si * rep, ..., sn)` 。输出的数据类型与 `x` 相同。