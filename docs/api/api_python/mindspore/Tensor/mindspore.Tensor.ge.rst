mindspore.Tensor.ge
===================

.. py:method:: mindspore.Tensor.ge(x)

    逐元素比较当前Tensor和参数 `x` 的大小。如果当前输入Tensor中的元素大于或等于参数 `x` 中对应元素，则对应位置返回True，否则返回False。

    更多参考详见 :func:`mindspore.ops.ge`。

    参数：
        - **x** (Tensor) - 输入Tensor。

    返回：
        Tensor，与广播后的输入shape相同，数据类型为bool。
    