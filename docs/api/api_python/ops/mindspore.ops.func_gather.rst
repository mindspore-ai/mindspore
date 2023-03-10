mindspore.ops.gather
======================

.. py:function:: mindspore.ops.gather(input_params, input_indices, axis, batch_dims=0)

    返回输入Tensor在指定 `axis` 上 `input_indices` 索引对应的元素组成的切片。

    下图展示了Gather常用的计算过程：

    .. image:: Gather.png

    其中，params代表输入 `input_params` ，indices代表要切片的索引 `input_indices` 。

    .. note::
        1. input_indices的值必须在 `[0, input_param.shape[axis])` 范围内，超出该范围结果未定义。
        2. Ascend平台上，input_params的数据类型当前不能是 `bool_ <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 。

    参数：
        - **input_params** (Tensor) - 原始Tensor，shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **input_indices** (Tensor) - 要切片的索引Tensor，shape为 :math:`(y_1, y_2, ..., y_S)` 。指定原始Tensor中要切片的索引。数据类型必须是int32或int64。
        - **axis** (int) - 指定要切片的维度索引。
        - **batch_dims** (int) - 指定batch维的数量。默认值：0。

    返回：
        Tensor，shape为 :math:`input\_params.shape[:axis] + input\_indices.shape[batch\_dims:] + input\_params.shape[axis + 1:]` 。

    异常：
        - **TypeError** - `axis` 不是int。
        - **TypeError** - `input_params` 不是Tensor。
        - **TypeError** - `input_indices` 不是int类型的Tensor。
