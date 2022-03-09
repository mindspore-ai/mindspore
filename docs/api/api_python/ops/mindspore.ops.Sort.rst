mindspore.ops.Sort
===================

.. py:class:: mindspore.ops.Sort(axis=-1, descending=False)

    根据指定的维度对输入Tensor的元素进行排序。默认为升序排序。

    **参数：**

    - **axis** (int) - 指定排序的轴。默认值：-1。
    - **descending** (bool) - 指定排序方式。如果descending为True，则根据value对元素进行升序排序。默认值：False。

    **输入：**

    - **x** (Tensor) - Sort的输入，任意维度的Tensor，数据类型为float16或float32。

    **输出：**

    - **y1** (Tensor) - Tensor，其值为排序后的值，shape和数据类型与输入相同。
    - **y2** (Tensor) - 输入Tensor，其元素的索引。数据类型为int32。

    **异常：**

    - **TypeError** - `axis` 不是int。
    - **TypeError** - `descending` 不是bool。
    - **TypeError** - `x` 的数据类型既不是float16也不是float32。