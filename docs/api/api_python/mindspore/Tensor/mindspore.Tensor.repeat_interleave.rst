mindspore.Tensor.repeat_interleave
===================================

.. py:method:: mindspore.Tensor.repeat_interleave(repeats, dims=None)

    沿着轴重复Tensor的元素，类似 `numpy.Repeat`。

    参数：
        - **repeats** (int) - 指定复制次数，为正数。
        - **dims** (int) - 指定复制轴，如果为None，则默认为0。

    返回：
        Tensor，值沿指定轴复制。如果输入的shape为 :math:`(s1, s2, ..., sn)` ，轴为i，则输出的shape为 :math:`(s1, s2, ..., si * repeats, ..., sn)` 。输出的数据类型与输入相同。
