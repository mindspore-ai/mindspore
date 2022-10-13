mindspore.Tensor.gather
=======================

.. py:method:: mindspore.Tensor.gather(input_indices, axis)

    返回指定 `axis` 上 `input_indices` 的元素对应的输入Tensor切片。输入Tensor的形状是 :math:`(x_1, x_2, ..., x_R)`。为了方便描述，对于输入Tensor记为 `input_params`。

    .. note::
        1. input_indices 的值必须在 `[0, input_params.shape[axis])` 的范围内，结果未定义超出范围。
        2. 当前在Ascend平台，input_params的值不能是 `bool_ <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.html#mindspore.dtype>`_ 类型。

    参数：
        - **input_indices** (Tensor) - 待切片的索引张量，其形状为 :math:`(y_1, y_2, ..., y_S)`，代表指定原始张量元素的索引，其数据类型包括：int32，int64。
        - **axis** (int) - 指定维度索引的轴以搜集切片。

    返回：
        Tensor，其中shape维度为 :math:`input\_params.shape[:axis] + input\_indices.shape + input\_params.shape[axis + 1:]`。

    异常：
        - **TypeError** - 如果 `axis` 不是一个整数。
        - **TypeError** - 如果 `input_indices` 不是一个整数类型的Tensor。