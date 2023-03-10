mindspore.ops.RealDiv
=======================

.. py:class:: mindspore.ops.RealDiv

    第一个输入Tensor元素为分子，第二个输入Tensor元素为分母，逐元素进行浮点型除法运算。

    更多参考详见 :func:`mindspore.ops.div`。

    .. math::
        out_{i} = input_{i} / other_{i}

    输入：
        - **x** (Union[Tensor, Number, bool]) - 第一个输入，为数值型，或bool，或数据类型为数值型或bool的Tensor。
        - **y** (Union[Tensor, Number, bool]) - 第二个输入，当第一个输入是Tensor时，第二个输入必须是一个数值型或bool，或是数据类型为数值型或bool的Tensor。

    输出：
        Tensor，输出的shape与广播后的shape相同，数据类型取两个输入中精度较高或数字较高的。
