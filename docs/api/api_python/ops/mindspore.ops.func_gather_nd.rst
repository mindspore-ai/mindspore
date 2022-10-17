mindspore.ops.gather_nd
=======================

.. py:function:: mindspore.ops.gather_nd(input_x, indices)

    根据索引获取输入Tensor指定位置上的元素。

    `indices` 是K维integer Tensor。假设 `indices` 是一个(K-1)维的张量，它的每个元素定义了 `input_x` 的一个slice：

    .. math::
        output[(i_0, ..., i_{K-2})] = input\_x[indices[(i_0, ..., i_{K-2})]]

    `indices` 的最后一维的长度不能超过 `input_x` 的秩： :math:`indices.shape[-1] <= input\_x.rank` 。

    参数：
        - **input_x** (Tensor) - GatherNd的输入。任意维度的Tensor。
        - **indices** (Tensor) - 索引Tensor，其数据类型为int32或int64。

    返回：
        Tensor，数据类型与 `input_x` 相同，shape为 `indices_shape[:-1] + input_x_shape[indices_shape[-1]:]` 。

    异常：
        - **ValueError** - `input_x` 的shape长度小于 `indices` 的最后一维的长度。
