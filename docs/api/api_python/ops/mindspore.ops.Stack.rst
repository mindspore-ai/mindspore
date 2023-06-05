mindspore.ops.Stack
====================

.. py:class:: mindspore.ops.Stack(axis=0)

    在指定轴上对输入Tensor序列进行堆叠。

    更多参考详见 :func:`mindspore.ops.stack`。

    参数：
        - **axis** (int，可选) - 指定堆叠运算的轴。取值范围为[-(R+1), R+1)。默认值： ``0`` 。

    输入：
        - **input_x** (Union[tuple, list]) - 输入多个Tensor对象组成的tuple或list，每个Tensor具有相同shape和数据类型。

    输出：
        堆叠运算后的Tensor，数据类型和 `input_x` 的相同。
