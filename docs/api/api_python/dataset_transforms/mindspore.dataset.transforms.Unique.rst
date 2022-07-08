mindspore.dataset.transforms.Unique
===================================

.. py:class:: mindspore.dataset.transforms.Unique()

    对输入张量进行唯一运算，每次只支持对一个数据列进行变换。

    Unique将返回3个Tensor: 运算结果Tensor，索引Tensor，计数Tensor。

    - 运算结果Tensor包含输入张量的所有唯一元素，且和输入张量的顺序是一样的。
    - 索引Tensor包含输入Tensor的每个元素在运算结果中的索引。
    - 计数Tensor包含运算结果Tensor的每个元素的计数。

    .. note:: 需要在 `batch` 操作之后调用该运算。

    异常：
        - **RuntimeError** - 当输入的Tensor具有两列。

