mindspore.ops.Equal
====================

.. py:class:: mindspore.ops.Equal

    逐元素比较两个输入Tensor是否相等。

    更多参考详见 :func:`mindspore.ops.equal`。

    输入：
        - **x** (Union[Tensor, Number]) - 第一个输入可以是数值型，也可以是数据类型为数值型的Tensor。
        - **y** (Union[Tensor, Number]) - 当第一个输入是Tensor时，第二个输入是数值型或数据类型为数值型的Tensor。数据类型与第一个输入相同。

    输出：
        Tensor，shape与输入 `x` 和 `y` 广播后的shape相同，数据类型为bool。
