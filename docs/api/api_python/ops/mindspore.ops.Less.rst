mindspore.ops.Less
===================

.. py:class:: mindspore.ops.Less

    逐元素计算 :math:`x < y` ，返回为bool。

    详情请查看 :func:`mindspore.ops.less` 。

    输入：
        - **x** (Union[Tensor, Number, bool]) - 第一个输入，为数值型，或bool，或数据类型为数值型或bool的Tensor。
        - **y** (Union[Tensor, Number, bool]) - 第二个输入，当第一个输入是Tensor时，第二个输入必须是一个数值型或bool，或是数据类型为数值型或bool的Tensor。

    输出：
        Tensor，输出shape与广播后的shape相同，数据类型为bool。
