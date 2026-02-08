mindspore.Tensor.nonzero
========================

.. py:method:: mindspore.Tensor.nonzero(*, as_tuple=False)

    返回所有非零元素下标位置。

    .. note::
        关于 `self` 的秩的限制。

        - Ascend：其秩可以等于0，GE后端除外。
        - CPU/GPU：其秩应大于等于1。

    关键字参数：
        - **as_tuple** (bool, 可选) - 是否以tuple形式输出。默认值： ``False`` ，输出Tensor。如果为 ``True`` ，输出tuple[Tensor]，只支持 ``Ascend`` 。

    返回：
        - 当 `as_tuple=False` ，输出Tensor，维度为2，类型为int64，表示输入中所有非零元素的下标。
        - 当 `as_tuple=True` ，输出tuple[Tensor]，类型为int64，长度为输入张量的维度，每一元素是输入张量在该维度下所有非零元素的下标的1D张量。

    异常：
        - **TypeError** - 如果 `self` 不是Tensor。
        - **TypeError** - 如果 `as_tuple` 不是bool。
        - **RuntimeError** - 在CPU或者GPU或者Ascend的GE后端中，如果 `self` 的维度为0。
