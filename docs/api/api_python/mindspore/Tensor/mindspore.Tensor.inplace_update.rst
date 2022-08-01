mindspore.Tensor.inplace_update
===============================

.. py:method:: mindspore.Tensor.inplace_update(v, indices)

    根据 `indices` 以 `v` 来更新Tensor中的值。

    .. note::
        `indices` 只能沿着最高轴进行索引。

    参数：
        - **v** (Tensor) - 用来更新的值。有相同的数据类型和除第一维外相同的shape。第一维的大小应该与 `indices` 大小相同。
        - **indices** (Union[int, tuple]) - 待更新值在原Tensor中的索引。

    返回：
        Tensor，更新后的Tensor。

    异常：
        - **TypeError** - `indices` 不是int或tuple。
        - **TypeError** - `indices` 是元组，但是其中的元素不是int。
        - **ValueError** - Tensor的shape与 `v` 的shape不同。