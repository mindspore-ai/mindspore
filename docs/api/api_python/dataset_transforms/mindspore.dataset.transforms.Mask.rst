mindspore.dataset.transforms.Mask
=================================

.. py:class:: mindspore.dataset.transforms.Mask(operator, constant, dtype=mstype.bool_)

    用给条件判断输入Tensor的内容，并返回一个掩码Tensor。Tensor中任何符合条件的元素都将被标记为True，否则为False。

    参数：
        - **operator** (:class:`mindspore.dataset.transforms.Relational`) - 关系操作符，可以取值为Relational.EQ、Relational.NE、Relational.LT、Relational.GT、Relational.LE、Relational.GE。以Relational.EQ为例，将找出Tensor中与 `constant` 相等的元素。
        - **constant** (Union[str, int, float, bool]) - 与输入Tensor进行比较的基准值。
        - **dtype** (:class:`mindspore.dtype`, 可选) - 生成的掩码Tensor的数据类型。默认值：mstype.bool\_ 。

    异常：
        - **TypeError** - 参数 `operator` 类型不为 :class:`mindspore.dataset.transforms.Relational` 。
        - **TypeError** - 参数 `constant` 类型不为str、int、float或bool。
        - **TypeError** - 参数 `dtype` 类型不为 :class:`mindspore.dtype` 。 
