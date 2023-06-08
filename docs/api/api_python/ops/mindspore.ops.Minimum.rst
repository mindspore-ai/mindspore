mindspore.ops.Minimum
======================

.. py:class:: mindspore.ops.Minimum

    逐元素计算两个Tensor的最小值。

    详情请查看 :func:`mindspore.ops.minimum` 。

    输入：
        - **x** (Union[Tensor, Number, bool]) - 第一个输入可以是Number或bool，也可以是数据类型为Number或bool的Tensor。
        - **y** (Union[Tensor, Number, bool]) - 第二个输入可以是数据类型为Number或bool的Tensor，当第一个输入是Tensor时，也可以是bool或Number。

    输出：
        一个Tensor，其shape与广播后的shape相同，其数据类型为两个输入中精度较高的类型。
