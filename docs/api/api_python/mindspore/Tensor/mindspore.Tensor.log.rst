mindspore.Tensor.log
====================

.. py:method:: mindspore.Tensor.log()

    逐元素返回当前Tensor的自然对数。

    .. math::
        y_i = log_e(x_i)

    .. note::
        Ascend上输入Tensor的维度要小于等于8，CPU上输入Tensor的维度要小于8。

    .. warning::
        如果算子Log的输入值在(0，0.01]或[0.95，1.05]范围内，则输出精度可能会存在误差。

    返回：
        Tensor，具有与当前Tensor相同的数据类型和shape。

    异常：
        - **TypeError** - 在GPU和CPU平台上运行时，当前Tensor的数据类型不为以下类型： float16、 float32、 float64、 complex64 和 complex128。
        - **TypeError** - 在Ascend平台上运行时，当前Tensor的数据类型不是float16或float32。
