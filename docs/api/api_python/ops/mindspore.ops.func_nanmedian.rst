mindspore.ops.nanmedian
=======================

.. py:function:: mindspore.ops.nanmedian(input, axis=-1, keepdims=False)

    计算 `input` 指定维度元素的中值和索引，忽略NaN。如果指定维度中的所有元素都是NaN，则结果将是NaN。

    .. warning::
        如果 `input` 的中值不唯一，则 `indices` 不一定包含第一个出现的中值。

    参数：
        - **input** (Tensor) - 计算中值和索引的输入Tensor。
        - **axis** (int, 可选) - 求中值和索引的维度。默认值：``-1`` ，计算最后一维。
        - **keepdims** (bool, 可选) - 输出Tensor是否保持维度。默认值： ``False`` ，不保留维度。

    返回：
        - **y** (Tensor) - 指定维度输入的中值，数据类型与 `input` 相同。
        - **indices** (bool) - 中值索引。数据类型为int64。

    异常：
        - **TypeError** - `input` 的数据类型不是: int16、int32、int64、float32、float64。
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `axis` 不是int类型。
        - **TypeError** - `keepdims` 不是bool类型。
        - **ValueError** - `axis` 的范围不在[-r, r)，`r` 表示 `input` 的rank。
