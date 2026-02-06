mindspore.Tensor.index_add
==========================

.. py:method:: mindspore.Tensor.index_add(indices, y, axis, use_lock=True, check_index_bound=True) -> Tensor

    将Tensor `y` 加到Parameter或Tensor `self` 的指定 `axis` 轴和指定 `indices` 位置。要求 `axis` 轴的取值范围
    为[0, len(self.dim) - 1]， `indices` 中元素的取值范围为[0, self.shape[axis] - 1]。

    参数：
        - **indices** (Tensor) - 指定Tensor `y` 加到 `self` 的 `axis` 轴的指定下标位置，要求数据类型为int32。
          要求 `indices` shape的维度为一维，并且 `indices` shape的大小与 `y` shape在 `axis` 轴上的大小一致。 `indices` 中元素
          取值范围为[0, b)，其中b的值为 `self` shape在 `axis` 轴上的大小。
        - **y** (Tensor) - 与 `self` 相加的Tensor。
        - **axis** (int) - 指定沿哪根轴相加。
        - **use_lock** (bool，可选) - 是否对参数更新过程加锁保护。如果为 ``True`` ，在更新参数 `self` 的值时使用原子操作以实现加锁保护，如果为
          ``False`` ， `self` 的值可能会不可预测。默认值： ``True`` 。
        - **check_index_bound** (bool，可选) - ``True`` 表示检查 `indices` 边界， ``False`` 表示不检查。默认值： ``True`` 。

    返回：
        相加后的Tensor。shape和数据类型与输入 `self` 相同。

    异常：
        - **TypeError** - `indices` 或者 `y` 的类型不是Tensor。
        - **ValueError** - `axis` 的值超出 `self` shape的维度范围。
        - **ValueError** - `self` shape的维度和 `y` shape的维度不一致。
        - **ValueError** - `indices` shape的维度不是一维，或者 `indices` shape的大小与 `y` shape在 `axis` 轴上的大小不一致。
        - **ValueError** - 除 `axis` 轴外， `self` shape和 `y` shape的大小不一致。

    .. py:method:: mindspore.Tensor.index_add(dim, index, source, *, alpha=1) -> Tensor
        :noindex:

    详情请参考 :func:`mindspore.ops.index_add`。
    其中 `Tensor.index_add` 与 :func:`mindspore.ops.index_add` 参数对应关系如下：
    `dim` -> `axis` 、 `index` -> `indices` 、 `source * alpha` -> `y`。
