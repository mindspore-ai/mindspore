mindspore.ops.repeat_interleave
================================

.. py:function:: mindspore.ops.repeat_interleave(input, repeats, axis=None)

    沿着轴重复Tensor的元素，类似 `numpy.Repeat`。

    参数：
        - **input** (Tensor) - 进行重复操作的入参Tensor，类型必须为float16，float32，int8，uint8，int16，int32或者int64。
        - **repeats** (int) - 指定复制次数，为正数。
        - **axis** (int，可选) - 指定复制轴，默认值：None。如果为None，输入Tensor会被展平并且输出结果也会被展平。

    返回：
        Tensor，值沿指定轴复制。如果输入的shape为 :math:`(s1, s2, ..., sn)` ，轴为i，则输出的shape为 :math:`(s1, s2, ..., si * repeats, ..., sn)` 。输出的数据类型与输入相同。
