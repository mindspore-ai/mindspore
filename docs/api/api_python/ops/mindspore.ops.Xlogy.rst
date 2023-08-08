mindspore.ops.Xlogy
====================

.. py:class:: mindspore.ops.Xlogy

    计算第一个输入Tensor乘以第二个输入Tensor的对数。当 `x` 为零时，则返回零。

    更多参考详见 :func:`mindspore.ops.xlogy`。

    输入：
        - **x** (Union[Tensor, number.Number, bool]) - 第一个输入为数值型。数据类型为 `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ 或 `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.dtype>`_ 。
        - **y** (Union[Tensor, number.Number, bool]) - 第二个输入为数值型。当第一个输入是Tensor或数据类型为数值型或bool的Tensor时， 则第二个输入是数值型或bool。当第一个输入是Scalar时，则第二个输入必须是数据类型为数值型或bool的Tensor。

    输出：
        Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高或数数值较高的类型。
