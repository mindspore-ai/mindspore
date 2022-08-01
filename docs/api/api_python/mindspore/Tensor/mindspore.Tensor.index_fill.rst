mindspore.Tensor.index_fill
===========================

.. py:method:: mindspore.Tensor.index_fill(dim, index, value)

    按 `index` 中给定的顺序选择索引，将输入 `value` 值填充到当前Tensor的所有 `dim` 维元素。

    参数：
        - **dim** (Union[int, Tensor]) - 填充输入Tensor的维度，要求是一个int或者数据类型为int32或int64的0维Tensor。
        - **index** (Tensor) - 填充输入Tensor的索引，数据类型为int32。
        - **value** (Union[bool, int, float, Tensor]) - 填充输入Tensor的值。如果 `value` 是Tensor，那么 `value` 要求是数据类型与当前Tensor相同的0维Tensor。否则，该值会自动转化为一个数据类型与当前Tensor相同的0维Tensor。

    返回：
        填充后的Tensor。shape和数据类型与当前Tensor相同。

    异常：
        - **TypeError** - `dim` 的类型不是int或者Tensor。
        - **TypeError** - 当 `dim` 是Tensor时， `dim` 的数据类型不是int32或者int64。
        - **TypeError** - `index` 的类型不是Tensor。
        - **TypeError** - `index` 的数据类型不是int32。
        - **TypeError** - `value` 的类型不是bool、int、float或者Tensor。
        - **TypeError** - 当 `value` 是Tensor时， `value` 的数据类型和当前Tensor的数据类型不相同。
        - **ValueError** - 当 `dim` 是Tensor时， `dim` 的维度不等于0。
        - **ValueError** - `index` 的维度大于1。
        - **ValueError** - 当 `value` 是Tensor时， `value` 的维度不等于0。
        - **RuntimeError** - `dim` 值超出范围[-self.ndim, self.ndim - 1]。
        - **RuntimeError** - `index` 存在值超出范围[-self.shape[dim], self.shape[dim]-1]。