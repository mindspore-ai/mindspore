mindspore.ops.GreaterEqual
===========================

.. py:class:: mindspore.ops.GreaterEqual

    输入两个Tensor，逐元素比较第一个Tensor是否大于等于第二个Tensor。

    更多参考详见 :func:`mindspore.ops.ge`。

    输入：
        - **x** (Union[Tensor, Number, bool]) - 第一个输入可以是Number，也可以是数据类型为Number的Tensor。
        - **y** (Union[Tensor, Number, bool]) - 第二个输入是Number，当第一个输入是Tensor时，也可以是bool，或数据类型为Number或bool的Tensor。

    输出：
        Tensor，输出的shape与输入广播后的shape相同，数据类型为bool。
