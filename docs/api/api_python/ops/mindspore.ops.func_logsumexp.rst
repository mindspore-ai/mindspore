mindspore.ops.logsumexp
=======================

.. py:function:: mindspore.ops.logsumexp(input, axis, keep_dims=False)

    求Tensor的对数指数和。

    .. math::

        logsumexp(input) = \log(\sum e^{input-input_{max}}) + input_{max}

    .. note::
        Ascend上输入Tensor的维度要小于等于8，CPU上输入Tensor的维度要小于8。

    参数：
        - **input** (Tensor) - 任意维度的输入Tensor。数据类型为float16或float32。
        - **axis** (Union[int, tuple(int), list(int)]) - 需要reduce的维度，输入为 `()` 时reduce所有维度。
        - **keep_dims** (bool) - 是否保留reduce的维度。若设为True，则计算后指定的axis的长度为1，否则不保留该维度。默认值：False。

    返回：
        Tensor，具有与 `input` 相同的数据类型。

        - 若axis为()，且 `keep_dims` 设为False，则输出为零维Tensor。
        - 若axis为int类型，假设为2，且 `keep_dims` 设为False，则输出的shape为 :math:`(input_1, input_3, ..., input_R)` 。
        - 若axis为tuple(int)，假设为(2, 3)，且 `keep_dims` 为False，则输出shape为 :math:`(input_1, input_4, ..., input_R)` 。
