mindspore.ops.Maximum
======================

.. py:class:: mindspore.ops.Maximum

    逐元素计算输入Tensor的最大值。

    更多参考详见 :func:`mindspore.ops.maximum`。

    输入：
        - **x** (Union[Tensor, Number, bool]) - 第一个输入可以是Number或bool，也可以是数据类型为Number或bool的Tensor。
        - **y** (Union[Tensor, Number, bool]) - 第二个输入是Number，当第一个输入是Tensor时，也可以是bool，或数据类型为Number或bool的Tensor。

    输出：
        Tensor的shape与广播后的shape相同，数据类型为两个输入中精度较高或数字较多的类型。
