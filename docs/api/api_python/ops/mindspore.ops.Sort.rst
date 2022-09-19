mindspore.ops.Sort
===================

.. py:class:: mindspore.ops.Sort(axis=-1, descending=False)

    根据指定的轴对输入Tensor的元素进行排序，默认为升序排序。

    .. warning::
        目前仅支持float16数据类型。如果使用float32类型可能导致数据精度损失。

    参数：
        - **axis** (int) - 指定排序的轴。默认值：-1。
        - **descending** (bool) - 指定排序方式。如果 `descending` 为True，则根据value对元素进行降序排序。默认值：False。

    输入：
        - **x** (Tensor) - Sort的输入，任意维度的Tensor，数据类型为float16或float32。

    输出：
        - **y1** (Tensor) - Tensor，其值为排序后的值，shape和数据类型与输入相同。
        - **y2** (Tensor) - 输入Tensor，其元素的索引。数据类型为int32。

    异常：
        - **TypeError** - `axis` 不是int。
        - **TypeError** - `descending` 不是bool。
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。
        - **ValueError** - 当 `axis` 取值不在[-len(x.shape), len(x.shape))范围内。
