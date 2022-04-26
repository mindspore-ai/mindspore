mindspore.nn.Roll
=================

.. py:class:: mindspore.nn.Roll(shift, axis)

    沿轴移动Tensor的元素。

    元素沿着 `axis` 维度按照 `shift` 偏移（朝着较大的索引）正向移动。 `shift` 为负值则使元素向相反方向移动。移动最后位置的元素将绕到第一个位置，反之亦然。可以指定沿多个轴的多个偏移。

    **参数：**

    - **shift** (Union[list(int), tuple(int), int]) - 指定元素移动方式，如果为整数，则元素沿指定维度正向移动（朝向较大的索引）的位置数。负偏移将向相反的方向滚动元素。
    - **axis** (Union[list(int), tuple(int), int]) - 指定需移动维度的轴。

    **输入：**

    - **input_x** (Tensor) - 输入Tensor。

    **输出：**

    Tensor，shape和数据类型与输入的 `input_x` 相同。

    **异常：**

    - **TypeError** - `shift` 不是int、tuple或list。
    - **TypeError** - `axis` 不是int、tuple或list。
    - **TypeError** - `shift` 的元素不是int。
    - **TypeError** - `axis` 的元素不是int。
    - **ValueError** - `axis` 超出[-len(input_x.shape), len(input_x.shape))范围。
    - **ValueError** - `shift` 的shape长度不等于 `axis` 的shape长度。