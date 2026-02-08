mindspore.Tensor.view
=====================

.. py:method:: mindspore.Tensor.view(*shape) -> Tensor

    根据输入shape重新创建一个tensor，与原tensor数据相同。

    参数：
        - **shape** (Union[tuple(int), int]) - 输出tensor的维度。

    返回：
        Tensor，具有与入参 `shape` 相同的维度。

    .. py:method:: mindspore.Tensor.view(dtype) -> Tensor
        :noindex:

    返回一个与原tensor数据相同但数据类型不同的新tensor。

    .. note::

        - 如果 `dtype` 的元素大小不同于原tensor数据类型的元素大小，则原tensor必须同时满足以下条件：

          - 原tensor的shape不能为空，这意味着原tensor不能是一个标量tensor。
          - 原tensor的最后一个步幅必须为 ``1`` 。

        - 如果 `dtype` 的元素大小大于原tensor数据类型的元素大小，则原tensor还必须满足以下条件：

          - 原tensor的最后一个维度必须能被 `dtype` 的元素大小与原tensor的数据类型的元素大小之比整除。
          - 原tensor的内存偏移量必须能被 `dtype` 的元素大小与原tensor的数据类型的元素大小之比整除。
          - 除最后一个维度外，所有维度的步幅必须能被 `dtype` 的元素大小与原tensor的数据类型的元素大小之比整除。

        - 仅支持PyNative模式。

    参数：
        - **dtype** (:class:`mindspore.dtype`) - 指定的返回tensor的数据类型。

    返回：
        Tensor，其数据与原tensor的数据相同，且数据类型为指定的数据类型。
