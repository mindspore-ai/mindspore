mindspore.ops.div
=================

.. py:function:: mindspore.ops.div(input, other, *, rounding_mode=None)

    逐元素计算第一个输入Tensor除以第二输入Tensor的商。

    `input` 和 `other` 的输入遵循隐式类型转换规则，使数据类型一致。
    输入必须是两个Tensor，或一个Tensor和一个Scalar。
    当输入是两个Tensor时，它们的数据类型不能同时为bool，它们的shape可以广播。
    当输入是一个Tensor和一个Scalar时，Scalar只能是一个常量。

    .. math::
        out_{i} = input_{i} / other_{i}

    参数：
        - **input** (Union[Tensor, Number, bool]) - 第一个输入，为数值型，或bool，或数据类型为数值型或bool的Tensor。
        - **other** (Union[Tensor, Number, bool]) - 第二个输入，当第一个输入是Tensor时，第二个输入必须是一个数值型或bool，或是数据类型为数值型或bool的Tensor。

    关键字参数：
        - **rounding_mode** (str, 可选) - 应用于结果的舍入类型。三种类型被定义为None、"floor" 和 "trunc" 。默认值：None。

          - **None**: 默认行为。相当于Python中的 `true division` 或NumPy中的 `true_divide` 。
          - **"floor"**: 将除法的结果向下舍入。相当于Python中的 `floor division` 或NumPy中的 `floor_divide` 。
          - **"trunc"**: 将除法的结果舍入到零。相当于C语言风格的整数除法。

    返回：
        Tensor，输出的shape与广播后的shape相同，数据类型取两个输入中精度较高或数字较高的。

    异常：
        - **TypeError** - 如果 `input` 和 `other` 不是以下之一：Tensor、Number、bool。
        - **ValueError** - 如果 `rounding_mode` 不是以下之一：None、"floor"、"trunc"。
