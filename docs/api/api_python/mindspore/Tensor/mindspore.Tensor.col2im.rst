mindspore.Tensor.col2im
=======================

.. py:method:: mindspore.Tensor.col2im(output_size, kernel_size, dilation, padding_value, stride)

    将一组滑动的局部块组合成一个大张量。

    参数：
        - **output_size** (Tensor) - 输出张量的后两维的shape。
        - **kernel_size** (Union[int, tuple[int], list[int]]) - 滑动窗口的大小。
        - **dilation** (Union[int, tuple[int], list[int]]) - 滑动窗口扩张的大小。
        - **padding_value** (Union[int, tuple[int], list[int]]) - 填充的大小。
        - **stride** (Union[int, tuple[int], list[int]]) - 步长的大小。

    返回：
        Tensor，输出的张量，维度和类型和输入一致。

    异常：
        - **TypeError** - 如果 `kernel_size`，`dilation`，`padding_value`，`stride` 不属于 Union[int, tuple[int], list[int]]。
        - **ValueError** - 如果 `kernel_size`，`dilation`，`stride` 值小于等于0或者个数大于2。
        - **ValueError** - 如果 `padding_value` 值小于0或者个数大于2。