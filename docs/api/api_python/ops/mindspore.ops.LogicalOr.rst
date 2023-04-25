mindspore.ops.LogicalOr
========================

.. py:class:: mindspore.ops.LogicalOr

    逐元素计算两个Tensor的逻辑或运算。

    详情请查看 :func:`mindspore.ops.logical_or` 。

    输入：
        - **x** (Union[Tensor, bool]) - 第一个输入是bool或数据类型可被隐式转换为bool的Tensor。
        - **y** (Union[Tensor, bool]) - 当第一个输入是Tensor的时候，第二个输入是bool或者数据类型可被隐式转换为bool的Tensor。

    输出：
        Tensor，其shape与 `x` 和 `y` 广播后的shape相同，数据类型为bool。
