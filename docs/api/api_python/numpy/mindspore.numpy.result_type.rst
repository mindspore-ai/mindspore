mindspore.numpy.result_type
===========================

.. py:function:: mindspore.numpy.result_type(*arrays_and_dtypes)

    返回对入参使用类型提升规则所得的类型。

    .. note::
        提升规则与原版 NumPy 略有不同，更类似于 JAX，因为更倾向于 32 位而不是 64 位数据类型。复数类型不支持。

    参数：
        - **\*arrays_and_dtypes** (Union[int, float, bool, list, tuple, Tensor, mindspore.dtype, str]) - 需要得到类型结果的操作数。

    返回：
        mindspore.dtype，结果类型。

    异常：
        - **TypeError** - 如果输入不是有效的数据类型。