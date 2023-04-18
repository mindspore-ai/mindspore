mindspore.ops.NotEqual
========================

.. py:class:: mindspore.ops.NotEqual

    逐元素计算两个Tensor是否不相等。

    更多参考详见 :func:`mindspore.ops.ne`。

    输入：
        - **x** (Union[Tensor, Number, bool]) - 第一个输入可以是数值型或bool，也可以是数据类型为数值型或bool的Tensor。
        - **y** (Union[Tensor, Number, bool]) - 第二个输入可以是数值型或bool。也可以是数据类型为数值型或bool的Tensor。

    输出：
        Tensor，shape与输入 `x` 和 `y` 广播后的shape相同，数据类型为bool。
