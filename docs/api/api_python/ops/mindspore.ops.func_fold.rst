mindspore.ops.fold
====================

.. py:function:: mindspore.ops.fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)

    将提取出的滑动局部区域块还原成更大的输出Tensor。

    .. warning::
        目前，仅支持输出为一个四维的Tensor（类似图片格式）。

    参数：
        - **input** (Tensor) - 四维Tensor，支持数据类型为float16和float32。
        - **output_size** (Tensor) - 一维Tensor，包含两个元素，均为整数类型。
        - **kernel_size** (Union[int, tuple[int], list[int]]) - 滑窗大小。如果是两个int，则分别为滑窗的高度和宽度；如果是一个int，则高度和宽度均为这个int值。是一个必要参数。
        - **dilation** (Union[int, tuple[int], list[int]]，可选) - 窗口的空洞卷积的扩充率，如果是两个int，则分别作用于滑窗的高度和宽度；如果是一个int，则这个值作用于滑窗的高度和宽度。默认值：1。
        - **padding** (Union[int, tuple[int], list[int]]，可选) - 滑窗的隐式零填充量。如果是两个int，则分别为滑窗的高度和宽度方向的填充量；如果是一个int，则高度和宽度方向的填充量均为这个int值。默认值：0。
        - **stride** (Union[int, tuple[int], list[int]]，可选) - 空间维度上滑动的步长，如果是两个int，则分别为滑窗的高度和宽度方向上的步长；如果是一个int，则高度和宽度方向上的步长均为这个int值。默认值：1。

    返回：
        Tensor，数据类型与 `input` 相同，Tensor格式为：(N, C, H, W)。

    异常：
        - **TypeError** - 如果 `kernel_size` 、 `stride` 、 `dilation` 、 `kernel_size` 的数据类型不是int、tuple或者list。
        - **ValueError** - 如果 `kernel_size`, `dilation`, `stride` 包含元素的值小于等于0或者元素个数大于 `2` 。
        - **ValueError** - 如果 `padding` 包含元素的值小于零。
        - **ValueError** - 如果 `input.shape[2] != kernel_size[0] * kernel_size[1]`。
        - **ValueError** - 如果 `input.shape[3]` 与计算出的滑块数量不匹配。
