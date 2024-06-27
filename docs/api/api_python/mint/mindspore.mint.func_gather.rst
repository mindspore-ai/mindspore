mindspore.mint.gather
=====================

.. py:function:: mindspore.mint.gather(input, dim, index)

    返回输入Tensor在指定 `index` 索引对应的元素组成的切片。

    .. math::
        output[(i_0, i_1, ..., i_{dim}, i_{dim+1}, ..., i_n)] = input[(i_0, i_1, ..., index[(i_0, i_1, ..., i_{dim}, i_{dim+1}, ..., i_n)], i_{dim+1}, ..., i_n)]

    .. warning::
        在Ascend后端，以下场景将导致不可预测的行为：

        - 正向执行流程中，当 `index` 的取值不在范围 `[-input.shape[dim], input.shape[dim])` 内；
        - 反向执行流程中，当 `index` 的取值不在范围 `[0, input.shape[dim])` 内。

    参数：
        - **input** (Tensor) - 待索引切片取值的原始Tensor。
        - **dim** (int) - 指定要切片的维度索引。取值范围 `[-input.rank, input.rank)`。
        - **index** (Tensor) - 指定原始Tensor中要切片的索引。数据类型必须是int32或int64。需要同时满足以下条件：

          - `index.rank == input.rank`；
          - 对于 `axis != dim` ， `index.shape[axis] <= input.shape[axis]` ；
          - `index` 的取值在有效区间 `[-input.shape[dim], input.shape[dim])` ；

    返回：
        Tensor，数据类型与 `input` 保持一致，shape与 `index` 保持一致。

    异常：
        - **ValueError** - `input` 的shape取值非法。
        - **ValueError** - `dim` 取值不在有效范围 `[-input.rank, input.rank)`。
        - **ValueError** - `index` 的值不在有效范围。
        - **TypeError** - `index` 的数据类型非法。
