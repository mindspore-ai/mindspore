mindspore.Tensor.index_copy\_
=============================

.. py:method:: mindspore.Tensor.index_copy_(dim, index, tensor) -> Tensor

    根据 `index` 中的索引顺序，将 `tensor` 的元素复制到 `self` 中。

    .. note::
        `index` 的值必须在 :math:`[0, self.shape[dim])` 范围内，如果超出该范围，结果未定义。

        如果 `index` 的值包含重复的索引，则结果是不确定的，因为它取决于最后发生的拷贝操作。

    参数：
        - **dim** (int) - 指定 `index` 属于哪个维度。
        - **index** (Tensor) - 一维Tensor，其值为在 `self` 中沿指定的 `dim` 的索引。
        - **tensor** (Tensor) - 包含待复制元素的tensor。

    返回：
        Tensor `self` 。
