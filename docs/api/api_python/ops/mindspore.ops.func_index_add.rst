mindspore.ops.index_add
=======================

.. py:function:: mindspore.ops.index_add(x, indices, y, axis, use_lock=True, check_index_bound=True)

    将Tensor `y` 加到Parameter `x` 的指定 `axis` 轴的指定 `indices` 位置。要求 `axis` 轴的取值范围
    为[0, len(x.dim) - 1]， `indices` 中元素的取值范围为[0, x.shape[axis] - 1]。

    参数：
        - **x** (Parameter) - 被加的Parameter。
        - **indices** (Tensor) - 指定Tensor `y` 加到 `x` 的 `axis` 轴的指定下标位置，要求数据类型为int32。
          要求 `indices` shape的维度为一维，并且 `indices` shape的大小与 `y` shape在 `axis` 轴上的大小一致。 `indices` 中元素
          取值范围为[0, b)，其中b的值为 `x` shape在 `axis` 轴上的大小。
        - **y** (Tensor) - 与 `x` 加的Tensor。
        - **axis** (int) - 指定沿哪根轴相加。
        - **use_lock** (bool) - 是否对参数更新过程加锁保护。如果为True，在更新参数 `x` 的值时使用原子操作以实现加锁保护，如果为
          False， `x` 的值可能会不可预测。默认值： ``True`` 。
        - **check_index_bound** (bool) - True表示检查 `indices` 边界，False表示不检查。默认值： ``True`` 。

    返回：
        相加后的Tensor。shape和数据类型与输入 `x` 相同。

    异常：
        - **TypeError** - `x` 的类型不是Parameter。
        - **TypeError** - `indices` 或者 `y` 的类型不是Tensor。
        - **ValueError** - `axis` 的值超出 `x` shape的维度范围。
        - **ValueError** - `x` shape的维度和 `y` shape的维度不一致。
        - **ValueError** - `indices` shape的维度不是一维或者 `indices` shape的大小与 `y` shape在 `axis` 轴上的大小不一致。
        - **ValueError** - 除 `axis` 轴外，`x` shape和 `y` shape的大小不一致。
