mindspore.Tensor.put\_
======================

.. py:method:: mindspore.Tensor.put_(index, source, accumulate=False)

    将元素从 `source` 复制到 `index` 指定的位置， `index` 和 `source` 需要具有相同数量的元素，但不一定具有相同的形状。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **index** (LongTensor) - 张量中需要操作的索引。
        - **source** (Tensor) - 包含要从中复制的值的张量。
        - **accumulate** (bool, 可选) - 是否与自己累加，默认 ``False``。

    返回：
        Tensor，与输入Tensor具有相同的shape。

    异常：
        - **TypeError** - `index` 不是Long类型。
        - **TypeError** - `source` 和input类型不同。
