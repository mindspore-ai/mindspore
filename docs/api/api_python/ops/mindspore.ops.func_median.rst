mindspore.ops.median
====================

.. py:function:: mindspore.ops.median(x, global_median=False, axis=0, keep_dims=False)

    输出张量指定维度上的中值。

    .. warning::
        如果 `global_median` 为True，则 `indices` 无意义。

    参数：
        - **x** (Tensor) - median的输入，任意维度的Tensor。数据类型支持int16、int32、int64、float32或float64。
        - **global_median** (bool) - 表示是否对x的全部元素取中值。默认值：False。
        - **axis** (int，可选) - 指定计算维度。默认值：0。取值范围为[-dims, dims - 1]，`dims` 表示 `x` 的维度长度。
        - **keep_dims** (bool) - 表示是否减少维度，如果为True，输出将与输入保持相同的维度；如果为False，输出将减少维度。默认值：False。

    返回：
        - **y** (Tensor) - 返回指定维度上的中值，数据类型与 `x` 相同。
        - **indices** (bool) - 指定中值索引。数据类型为int64。如果 `global_median` 为True，则结果无意义。

    异常：
        - **TypeError** - `x` 的数据类型不是: int16, int32, int64, float32, float64。
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `global_median` 不是bool。
        - **TypeError** - `axis` 不是int。
        - **TypeError** - `keep_dims` 不是bool。
        - **ValueError** - `axis` 的范围不在[-dims, dims - 1]，`dims` 表示 `x` 的维度长度。
