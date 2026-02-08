mindspore.Tensor.set_const_arg
==============================

.. py:method:: mindspore.Tensor.set_const_arg(const_arg=True)

    指定该Tensor在作为网络入参时是否是一个常量。

    参数：
        - **const_arg** (bool，可选) - Tensor在作为网络入参时是否是一个常量。默认值： ``True`` 。

    返回：
        Tensor，被指定了是否是一个常量网络入参。

    异常：
        - **TypeError** - 如果 `const_arg` 不是一个布尔值。
