mindspore.ops.Roll
===================

.. py:class:: mindspore.ops.Roll(shift, axis)

    沿轴移动Tensor的元素。

    更多参考详见 :func:`mindspore.ops.roll`。

    参数：
        - **shift** (Union[list(int), tuple(int), int]) - 指定元素移动方式，如果为正数，则元素沿指定维度正向移动（朝向较大的索引）的位置数。负偏移将向相反的方向滚动元素。
        - **axis** (Union[list(int), tuple(int), int]) - 指定需移动维度的轴。

    输入：
        - **input_x** (Tensor) - 输入Tensor。

    输出：
        Tensor，shape和数据类型与输入 `x` 相同。
