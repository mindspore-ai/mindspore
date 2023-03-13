mindspore.ops.median
====================

.. py:function:: mindspore.ops.median(input, axis=-1, keepdims=False)

    输出Tensor指定维度上的中值与索引。

    参数：
        - **input** (Tensor) - median的输入，任意维度的Tensor。数据类型支持int16、int32、int64、float32或float64。
        - **axis** (int，可选) - 指定计算维度。默认值：-1。取值范围为[-dims, dims - 1]，`dims` 表示 `input` 的维度长度。
        - **keepdims** (bool，可选) - 表示是否减少维度，如果为True，输出将与输入保持相同的维度；如果为False，输出将减少维度。默认值：False。

    返回：
        - **y** (Tensor) - 返回指定维度上的中值，数据类型与 `input` 相同。
        - **indices** (bool) - 指定中值索引。数据类型为int64。

    异常：
        - **TypeError** - `input` 的数据类型不是: int16、int32、int64、float32、float64。
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `axis` 不是int。
        - **TypeError** - `keepdims` 不是bool。
        - **ValueError** - `axis` 的范围不在[-dims, dims - 1]，`dims` 表示 `input` 的维度长度。
