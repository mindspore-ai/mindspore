mindspore.ops.expand
====================

.. py:function:: mindspore.ops.expand(input_x, size)

    返回一个当前张量的新视图，其中单维度扩展到更大的尺寸。

    .. note::
        将 `-1` 作为维度的 `size` 意味着不更改该维度的大小。张量也可以扩展到更大的维度，新的维度会附加在前面。对于新的维度，`size` 不能设置为-1。

    参数：
        - **input_x** (Tensor) - 输入Tensor，shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **size** (Tensor) - `input_x` 的新shape大小。

    返回：
        Tensor，其大小为 `size` 。

    异常：
        - **TypeError** - 如果 `input_x` 或者 `size` 不是Tensor。
        - **TypeError** - 如果 `size` 的数据类型不是int16、int32或int64。
        - **ValueError** - 如果 `size` 的长度小于 `input_x` shape的大小。
        - **ValueError** - 如果 `size` 不是一个1D Tensor。
        - **ValueError** - 如果 `size` 不等于 维度不为1的 `input_x` 的shape。
        - **ValueError** - 如果 `size` 小于0。
        - **ValueError** - 如果输出的元素数量超过1000000。
       
