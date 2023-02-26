mindspore.ops.unfold
====================

.. py:function:: mindspore.ops.unfold(input, kernel_size, dilation=1, padding=0, stride=1)

    通过从数据格式为（N，C，H，W）的Tensor中提取局部滑块并沿新的维度连接它们来重新排列输入Tensor。

    .. warning::
        目前，仅支持输入为一个四维的Tensor（类似图片格式）。

    参数：
        - **input** (Tensor) - 四维Tensor，支持所有实数数据类型。
        - **kernel_size** (Union[int, tuple[int], list[int]]) - 滑窗大小。如果是两个int，则分别为滑窗的高度和宽度；如果是一个int，则高度和宽度均为这个int值。是一个必要参数。
        - **dilation** (Union[int, tuple[int], list[int]]，可选) - 窗口的空洞卷积的扩充率，如果是两个int，则分别作用于滑窗的高度和宽度；如果是一个int，则这个值作用于化窗的高度和宽度。默认值：1。
        - **padding** (Union[int, tuple[int], list[int]]，可选) - 滑窗的隐式零填充量。为单个int或者包含一/二/四个整数的tuple/list。默认值: 0。
            
          - 如果是一个int, 则pad_height = pad_width。
          - 如果是两个int, 则pad_height = padding[0], pad_width = padding[1]。
          - 如果是四个int, 则padding = [pad_height_top, pad_height_bottom, pad_width_left, pad_width_right]。
        - **stride** (Union[int, tuple[int], list[int]]，可选) - 空间维度上滑动的步长，如果是两个int，则分别为滑窗的高和宽方向上的步长；如果是一个int，则高和宽方向上的步长均为这个int值。默认值: 1。

    返回：
        Tensor，数据类型与 `input` 相同。

    异常：
        - **TypeError** - 如果 `kernel_size` 、 `stride` 、 `dilation` 、 `kernel_size` 的数据类型不是int、tuple或者list。
        - **ValueError** - 如果 `kernel_size` 、 `dilation` 、 `stride` 包含元素的值小于等于0或者元素个数大于 `2` 。
        - **ValueError** - 如果 `padding` 包含元素的值小于零。
