mindspore.ops.col2im
====================

.. py:function:: mindspore.ops.col2im(input_x, output_size, kernel_size, dilation, padding_value, stride)

    将一组滑动局部块组合成一个大的Tensor。

    参数：
        - **input_x** (Tensor) - 四维Tensor，输入的批量的滑动局部块，数据类型支持float16和float32。
        - **output_size** (Tensor) - 包含两个int元素的一维Tensor，输出张量的后两维的shape。
        - **kernel_size** (Union[int, tuple[int], list[int]]) - 滑动窗口的大小。tuple的两个元素分别对应kernel的高度与宽度。如果为一个int则kernel的高度与宽度均为该值。
        - **dilation** (Union[int, tuple[int], list[int]]) - 滑动窗口扩张的大小。
        - **padding_value** (Union[int, tuple[int], list[int]]) - 填充的大小。
        - **stride** (Union[int, tuple[int], list[int]]) - 步长的大小。

    返回：
        Tensor，输出的张量，维度和类型和输入一致。

    异常：
        - **TypeError** - 如果 `kernel_size`，`dilation`，`padding_value`，`stride` 不属于 Union[int, tuple[int], list[int]]。
        - **ValueError** - 如果 `kernel_size`，`dilation`，`stride` 值小于等于0或者个数大于2。
        - **ValueError** - 如果 `padding_value` 值小于0或者个数大于2。
        - **ValueError** - 如果 `input_x.dims(2)` 不等于 `kernel_size[0] * kernel_size[1]` 。
        - **ValueError** - 如果 `input_x.dims(3)` 与计算出的滑动块数量不匹配。
