mindspore.Tensor.gather
=======================

.. py:method:: mindspore.Tensor.gather(dim, index) -> Tensor

    返回输入Tensor在指定 `index` 对应的元素组成的切片。公式如下：

    .. math::
        output[(i_0, i_1, ..., i_{dim}, i_{dim+1}, ..., i_n)] = input[(i_0, i_1, ..., index[(i_0, i_1, ..., i_{dim}, i_{dim+1}, ..., i_n)], i_{dim+1}, ..., i_n)]

    .. warning::
        在Ascend后端，以下场景将导致不可预测的行为：

        - 正向执行流程中，当 `index` 的取值不在范围 :math:`[-self.shape[dim], self.shape[dim])` 内；
        - 反向执行流程中，当 `index` 的取值不在范围 :math:`[0, self.shape[dim])` 内。

    参数：
        - **dim** (int) - 指定要切片的维度索引。取值范围 :math:`[-self.rank, self.rank)`。
        - **index** (Tensor) - 指定原始Tensor中要切片的索引。数据类型必须是int32或int64。需要同时满足以下条件：

          - :math:`index.rank == self.rank`；
          - 对于 :math:`axis != dim` ， :math:`index.shape[axis] <= self.shape[axis]` ；
          - `index` 的取值在有效区间 :math:`[-self.shape[dim], self.shape[dim])` ；

    返回：
        Tensor，数据类型与 `self` 保持一致，shape与 `index` 保持一致。

    异常：
        - **ValueError** - `index` 的shape取值非法。
        - **ValueError** - `dim` 取值不在有效范围 :math:`[-self.rank, self.rank)`。
        - **ValueError** - `index` 的值不在有效范围。
        - **TypeError** - `index` 的数据类型非法。

    .. py:method:: mindspore.Tensor.gather(input_indices, axis, batch_dims=0) -> Tensor
        :noindex:

    返回输入Tensor在指定 `axis` 上 `input_indices` 索引对应的元素组成的切片。

    下图展示了Gather常用的计算过程：

    .. image:: ../../images/Gather.png

    其中，params代表输入 `self` ，indices代表要切片的索引 `input_indices` 。

    .. note::
        - `input_indices` 的值必须在 :math:`[0, self.shape[axis])` 范围内。CPU与GPU平台越界访问将会抛出异常，Ascend平台越界访问的返回结果是未定义的。
        - Ascend平台上， `self` 的数据类型当前不能是 `bool <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 。

    参数：
        - **input_indices** (Tensor) - 要切片的索引Tensor，shape为 :math:`(y_1, y_2, ..., y_S)` 。指定原始Tensor中要切片的索引。数据类型必须是int32或int64。
        - **axis** (Union(int, Tensor[int])) - 指定要切片的维度索引。它必须要大于或等于 `batch_dims`。若 `axis` 为Tensor，其size必须为1。
        - **batch_dims** (int，可选) - 指定batch维的数量。它必须要小于或等于 `input_indices` 的rank。默认值： ``0`` 。

    返回：
        Tensor，shape为 :math:`self.shape[:axis] + self\_indices.shape[batch\_dims:] + self.shape[axis + 1:]` 。

    异常：
        - **TypeError** - `axis` 不是int或Tensor。
        - **ValueError** - `axis` 为Tensor时，size不为1。
        - **TypeError** - `input_indices` 不是int类型的Tensor。
        - **RuntimeError** - `input_indices` 在CPU或GPU平台超出 :math:`[0, self.shape[axis])` 范围。
