mindspore.dataset.vision.SlicePatches
=====================================

.. py:class:: mindspore.dataset.vision.SlicePatches(num_height=1, num_width=1, slice_mode=SliceMode.PAD, fill_value=0)

    在水平和垂直方向上将Tensor切片为多个块。适合于Tensor高宽较大的使用场景。如果将 `num_height` 和 `num_width` 都设置为 1，则Tensor将保持不变。输出Tensor的数量等于 num_height*num_width。

    参数：
        - **num_height** (int, 可选) - 垂直方向的切块数量。默认值：1。
        - **num_width** (int, 可选) - 水平方向的切块数量。默认值：1。
        - **slice_mode** (:class:`mindspore.dataset.vision.SliceMode` , 可选) - 表示填充或丢弃，它可以是 [SliceMode.PAD, SliceMode.DROP] 中的任何一个。默认值：SliceMode.PAD。
        - **fill_value** (int, 可选) - 如果 `slice_mode` 取值为 SliceMode.PAD，该值表示在右侧和底部的边界填充宽度（以像素数计）。 `fill_value` 取值必须在[0，255]范围内。默认值：0。

    异常：
        - **TypeError** - 当 `num_height` 不是int。
        - **TypeError** - 当 `num_width` 不是int。
        - **TypeError** - 当 `slice_mode` 的类型不为 :class:`mindspore.dataset.vision.SliceMode` 。
        - **TypeError** - 当 `fill_value` 不是int。
        - **ValueError** - 当 `num_height` 不为正数。
        - **ValueError** - 当 `num_width` 不为正数。
        - **ValueError** - 当 `fill_value` 不在 [0, 255]范围内。
        - **RuntimeError** - 如果输入的Tensor不是 <H, W> 或<H, W, C> 格式。
