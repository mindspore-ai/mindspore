mindspore.ops.roll
===================

.. py:function:: mindspore.ops.roll(input, shifts, dims=None)

    沿轴移动Tensor的元素。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **shifts** (Union[list(int), tuple(int), int]) - 指定元素移动方式，如果为正数，则元素沿指定维度正向移动（朝向较大的索引）的位置数。负偏移将向相反的方向滚动元素。
        - **dims** (Union[list(int), tuple(int), int], optional) - 指定需移动维度的轴。默认值：None。如果dims为None，则会将输入Tensor展平后再进行roll计算，然后将计算结果reshape为输入的shape。

    返回：
        Tensor，shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `shifts` 不是int、tuple或list。
        - **TypeError** - `dims` 不是int、tuple或list。