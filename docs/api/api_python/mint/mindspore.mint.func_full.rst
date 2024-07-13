mindspore.mint.full
===================

.. py:function:: mindspore.mint.full(size, fill_value, *, dtype=None)

    创建一个指定shape的Tensor，并用指定值填充。

    参数：
        - **size** (Union(tuple[int], list[int])) - 指定输出Tensor的shape。
        - **fill_value** (Union(number.Number, Tensor)) - 用来填充输出Tensor的值。可以是一个标量值、0-D的Tensor或只有单个元素的1-D的Tensor。

    关键字参数：
        - **dtype** (mindspore.dtype) - 指定输出Tensor的数据类型。数据类型只支持 `bool_` 和 `number` ，更多细节详见 :class:`mindspore.dtype` 。默认值： ``None`` 。

    返回：
        Tensor。

    异常：
        - **TypeError** - `size` 不是元组或列表。
        - **ValueError** - `size` 中包含小于0的成员。
