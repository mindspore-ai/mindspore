mindspore.ops.argmax
====================

.. py:function:: mindspore.ops.argmax(x, axis=-1, output_type=mstype.int32)

    返回输入Tensor在指定轴上的最大值索引。

    如果输入Tensor的shape为 :math:`(x_1, ..., x_N)` ，则输出Tensor的shape为 :math:`(x_1, ..., x_{axis-1}, x_{axis+1}, ..., x_N)` 。

    参数：
        - **x** (Tensor) - Argmax的输入，shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **axis** (int) - 指定Argmax计算轴。默认值：-1。
        - **output_type** (:class:`mindspore.dtype`) - 指定输出数据类型。默认值： `mindspore.dtype.int32` 。

    返回：
        Tensor，输出为指定轴上输入Tensor最大值的索引。

    异常：
        - **TypeError** - `axis` 不是int。
        - **TypeError** - `output_type` 既不是int32也不是int64。
