mindspore.ops.index_fill
========================

.. py:function:: mindspore.ops.index_fill(x, dim, index, value)

    按 `index` 中给定的顺序选择索引，将输入 `value` 值填充到输入Tensor `x` 的所有 `dim` 维元素。

    参数：
        - **x** (Tensor) - 输入Tensor，支持的数据类型是数值型和布尔型。
        - **dim** (Union[int, Tensor]) - 填充输入Tensor的维度，要求是一个int或者数据类型为int32或int64的零维Tensor。
        - **index** (Tensor) - 填充输入Tensor的索引，数据类型为int32。
        - **value** (Union[bool, int, float, Tensor]) - 填充输入Tensor的值。如果 `value` 是Tensor，那么 `value` 要求是数据类型与 `x` 相同的零维Tensor。否则，该值会自动转化为一个数据类型与 `x` 相同的零维Tensor。

    返回：
        填充后的Tensor。shape和数据类型与输入 `x` 相同。

    异常：
        - **TypeError** - `x` 的类型不是Tensor。
        - **TypeError** - `dim` 的类型不是int或者Tensor。
        - **TypeError** - 当 `dim` 是Tensor时， `dim` 的数据类型不是int32或者int64。
        - **TypeError** - `index` 的类型不是Tensor。
        - **TypeError** - `index` 的数据类型不是int32。
        - **TypeError** - `value` 的类型不是bool、int、float或者Tensor。
        - **TypeError** - 当 `value` 是Tensor时， `value` 的数据类型和 `x` 的数据类型不相同。
        - **ValueError** - 当 `dim` 是Tensor时， `dim` 的维度不等于0。
        - **ValueError** - `index` 的维度大于1。
        - **ValueError** - 当 `value` 是Tensor时， `value` 的维度不等于0。
        - **RuntimeError** - `dim` 值超出范围[-x.ndim, x.ndim - 1]。
        - **RuntimeError** - `index` 存在值超出范围[-x.shape[dim], x.shape[dim]-1]。
