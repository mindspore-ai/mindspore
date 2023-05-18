mindspore.nn.TimeDistributed
==============================

.. py:class:: mindspore.nn.TimeDistributed(layer, time_axis, reshape_with_axis=None)

    时间序列封装层。

    TimeDistributed是一个封装层，它允许将一个网络层应用到输入的每个时间切片。 `x` 至少是三维。执行中有两种情况。当提供 `reshape_with_axis` 时，选择reshape方法会更高效；否则，将使用沿 `time_axis` 划分输入的方法，这种方法更通用。比如，在处理BN时无法提供 `reshape_with_axis` 。

    参数：
        - **layer** (Union[Cell, Primitive]) - 需被封装的Cell或Primitive。
        - **time_axis** (int) - 指定各个时间切片上的轴。
        - **reshape_with_axis** (int) - 将使用 `time_axis` 调整该轴。默认值： ``None`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, T, *)` 的Tensor。其中 :math:`*` 表示任意数量的附加维度。

    输出：
        shape为 :math:`(N, T, *)` 的Tensor。

    异常：
        - **TypeError** - layer不是Cell或Primitive类型。
