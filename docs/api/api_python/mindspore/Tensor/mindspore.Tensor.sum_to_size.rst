mindspore.Tensor.sum_to_size
============================

.. py:method:: mindspore.Tensor.sum_to_size(*size)

    将原Tensor按照指定 `size` 进行求和。`size` 必须可以扩展到Tensor的shape。

    参数：
        - **size** (Union[tuple(int), int]) - 期望输出Tensor的shape。

    返回：
        Tensor，根据 `size` 对原Tensor进行求和的结果。

    异常：
        - **ValueError** - `size` 不能扩展成原Tensor的大小。
