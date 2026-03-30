mindspore.Tensor.assign_value
=============================

.. py:method:: mindspore.Tensor.assign_value(value)

    将另一个Tensor的值赋给当前Tensor。

    .. warning::
        此接口将在2.9.0（不含）之后版本中废弃，请使用 :func:`mindspore.Tensor.copy_` 替代。

    参数：
        - **value** (Tensor) - 用于赋值的Tensor。

    返回：
        Tensor，赋值后的Tensor。
