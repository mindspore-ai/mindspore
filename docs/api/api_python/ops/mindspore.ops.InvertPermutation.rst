mindspore.ops.InvertPermutation
================================

.. py:class:: mindspore.ops.InvertPermutation

    计算索引的逆置换。

    该算子主要用于计算索引的逆置换。 `input_x` 是一个一维的整数Tensor，一个以0开始的索引数组，并将每个值与其索引位置交换。换句话说，对于输出Tensor和输入 `input_x` ，依赖此计算方法 :math:`y[x[i]] = i, \quad i \in [0, 1, \ldots, \text{len}(x)-1]` 。

    .. note::
        这些值必须包括0。不能有重复的值，并且值不能为负值。

    输入：
        - **input_x** (Union(tuple[int], list[int])) - 输入由多个整数构造，即 :math:`(y_1, y_2, ..., y_S)` 代表索引。值必须包括0。不能有重复值或负值。只允许常量。最大值必须等于 `input_x` 的长度。
        
    输出：
        tuple[int]。输出的长度与 `input_x` 相同。
        
    异常：
        - **TypeError** - 如果 `input_x` 既不是tuple也不是list。
        - **TypeError** - 如果 `input_x` 的元素不是int。
