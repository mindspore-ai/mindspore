mindspore.ops.div
=================

.. py:function:: mindspore.ops.div(input, other, *, rounding_mode=None)

    逐元素计算第一个输入tensor除以第二个输入tensor。

    .. math::
        out_{i} = input_{i} / other_{i}

    .. note::
        - 两个输入tensor的shape须可以进行广播。
        - 两个输入tensor不能同时为bool类型。
        - 两个输入tensor遵循隐式类型转换规则，使数据类型保持一致。

    参数：
        - **input** (Union[Tensor, Number, bool]) - 第一个输入tensor。
        - **other** (Union[Tensor, Number, bool]) - 第二个输入tensor。

    关键字参数：
        - **rounding_mode** (str, 可选) - 应用于计算结果的舍入类型。默认 ``None`` 。三种类型被定义为：

          - **None**：相当于Python中的 `true division` 或NumPy中的 `true_divide` 。
          - **"floor"**：向下取整。相当于Python中的 `floor division` 或NumPy中的 `floor_divide` 。
          - **"trunc"**：向0取整。相当于C语言风格的整数除法。

    返回：
        Tensor
