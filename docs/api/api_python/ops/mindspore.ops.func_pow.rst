mindspore.ops.pow
==================

.. py:function:: mindspore.ops.pow(input, exponent)

    计算 `input` 中每个元素的 `exponent` 次幂。

    当 `exponent` 是Tensor时， `input` 和 `exponent` 的shape必须是可广播的。

    .. math::

        out_{i} = input_{i} ^{ exponent_{i}}

    参数：
        - **input** (Union[Tensor, Number]) - 第一个输入，是一个Number值或数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `bool_ <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 的Tensor。
        - **exponent** (Union[Tensor, Number]) - 第二个输入，是一个Number值或数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 或 `bool_ <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高的类型。
