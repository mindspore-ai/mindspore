mindspore.ops.roll
===================

.. py:function:: mindspore.ops.roll(x, shifts, dims)

    沿轴移动Tensor的元素。

    参数：
        - **x** (Tensor) - 输入Tensor。
        - **shifts** (Union[list(int), tuple(int), int]) - 指定元素移动方式，如果为正数，则元素沿指定维度正向移动（朝向较大的索引）的位置数。负偏移将向相反的方向滚动元素。
        - **dims** (Union[list(int), tuple(int), int]) - 指定需移动维度的轴。

    返回：
        Tensor，shape和数据类型与输入 `x` 相同。

    异常：
        - **TypeError** - `shifts` 不是int、tuple或list。
        - **TypeError** - `dims` 不是int、tuple或list。