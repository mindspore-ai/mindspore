mindspore.ops.Size
==================

.. py:class:: mindspore.ops.Size

    返回一个Scalar，类型为整数，表示输入Tensor的大小，即Tensor中元素的总数。

    更多参考详见 :func:`mindspore.ops.size`。

    输入：
        - **input_x** (Tensor) - 输入参数，shape为 :math:`(x_1, x_2, ..., x_R)` 。数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 。

    输出：
        int，表示 `input_x` 元素大小的Scalar。它的值为 :math:`size=x_1*x_2*...x_R` 。
