mindspore.Tensor.median
=======================

.. py:method:: mindspore.Tensor.median(global_median=False, axis=0, keep_dims=False)

    返回指定维度上的中值。

    参数：
        - **global_median** (bool) - 表示是否对当前Tensor的全部元素取中值。默认值：False。
        - **axis** (int) - 计算中值的维度。默认值：(0), 取值范围为[-ndim, ndim - 1]，'ndim' 表示当前Tensor的维度长度。
        - **keep_dims** (bool) - 表示是否减少维度，如果为True，输出将与输入保持相同的维度；如果为False，输出将减少维度。默认值：False。

    返回：
        - **y** (Tensor) - 返回指定维度上的中值，数据类型与当前Tensor相同。
        - **indices** (bool) - 指定中值索引。数据类型为int64。如果 `global_median` 为True，则结果无意义。

    异常：
        - **TypeError** - 当前Tensor的类型不是: int16, int32, int64, float32, float64。
        - **TypeError** - `global_median` 不是bool。
        - **TypeError** - `axis` 不是int。
        - **TypeError** - `keep_dims` 不是bool。
        - **ValueError** - `axis` 的范围不在[-ndim, ndim - 1]。

