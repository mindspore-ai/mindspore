mindspore.ops.Median
====================

.. py:class:: mindspore.ops.Median(global_median=False, axis=0, keep_dims=False, ignore_nan=False)

    输出Tensor指定维度 `axis` 上的中值与其对应的索引。如果 `global_median` 为True，则计算Tensor中所有元素的中值。

    .. warning::
        - 如果 `input` 的中值不唯一，则 `indices` 不一定包含第一个出现的中值。该算子的具体实现方式和后端类型相关，CPU和GPU的返回值可能不相同。
        - 如果 `global_median` 为 ``True`` , 第二个输出 `indices` 无意义。

    参数：
        - **global_median** (bool, 可选) - 是否计算Tensor中所有元素的中值。默认值： ``False`` 。
        - **axis** (int, 可选) - 进行中值计算的轴。默认值： ``0`` 。
        - **keep_dims** (bool, 可选) - 是否保留 `axis` 指定的维度。默认值： ``False`` 。
        - **ignore_nan** (bool, 可选) - 是否忽略输入Tensor中的NaN值。默认值： ``False`` 。

    输入：
        - **x** (Tensor) - 要计算中值的Tensor。

    输出：
        - **y** (Tensor) - 中值，数据类型与 `x` 相同。

          - 如果 `global_median` 为 ``True`` ， `y` 只有一个元素。
          - 如果 `keep_dims` 为 ``True`` , `y` 的shape除了在 `axis` 维度上为1外与 `x` 一致。
          - 其他情况下， `y` 比 `x` 缺少 `axis` 指定的维度。
          
        - **indices** (Tensor) - 中值的索引。shape与 `y` 一致，数据类型为int64。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `global_median` 、 `keep_dims` 或 `ignore_nan` 被指定了一个非bool值。
        - **TypeError** - `axis` 不是int。
        - **ValueError** - `axis` 不在 [-x.dim, x.dim-1] 范围内。

