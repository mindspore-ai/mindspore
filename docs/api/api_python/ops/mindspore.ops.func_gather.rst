mindspore.ops.gather
======================

.. py:function:: mindspore.ops.gather(input_params, input_indices, axis, batch_dims=0)

    返回输入tensor在指定轴及指定索引上对应元素的切片。

    .. warning::
        2.9.0（不含）之后版本中，该接口参数顺序和语义会发生变化。

    下图展示了Gather常用的计算过程：

    .. image:: ../images/Gather.png

    其中，params代表输入 `input_params` ，indices代表要切片的索引 `input_indices` 。

    .. note::
        - input_indices的值必须在 `[0, input_params.shape[axis])` 范围内。CPU与GPU平台越界访问将会抛出异常，Ascend平台越界访问的返回结果是未定义的。
        - Ascend平台上，input_params的数据类型不能是 `mindspore.bool <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 。
        - 返回tensor的shape为 :math:`input\_params.shape[:axis] + input\_indices.shape[batch\_dims:] + input\_params.shape[axis + 1:]` 。

    参数：
        - **input_params** (Tensor) - 输入tensor。
        - **input_indices** (Tensor) - 指定索引。
        - **axis** (Union(int, Tensor[int])) - 指定轴。
        - **batch_dims** (int) - batch维的数量。默认 ``0`` 。

    返回：
        Tensor
