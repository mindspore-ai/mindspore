mindspore.ops.GatherNd
=======================

.. py:class:: mindspore.ops.GatherNd

    根据索引获取输入数据指定位置上的元素。

    `indices` 是K维整型Tensor。可以看作是从 `input_x` 中取K-1维Tensor，每个元素都是一个切片：

    .. math::
        output[(i_0, ..., i_{K-2})] = input\_x[indices[(i_0, ..., i_{K-2})]]

    `indices` 的最后一个维度不能超过 `input_x` 的秩： :math:`indices.shape[-1] <= input\_x.rank` 。

    **输入：**

    - **input_x** (Tensor) - GatherNd的输入。任意维度的Tensor。
    - **indices** (Tensor) - 索引Tensor，其数据类型为int32或int64。

    **输出：**

    Tensor，数据类型与 `input_x` 相同，shape为indices_shape[:-1] + x_shape[indices_shape[-1]:]。

    **异常：**

    - **ValueError**  - `input_x` 的shape长度小于 `indices` 的最后一个维度。
    