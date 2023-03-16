mindspore.ops.ExtractVolumePatches
===================================

.. py:class:: mindspore.ops.ExtractVolumePatches(kernel_size, strides, padding)

    从输入中提取数据，并将它放入"depth"输出维度中，"depth"为输出的第二维。

    参数：
        - **kernel_size** (Union[int, tuple[int], list[int]]) - 长度为3或5的int列表。输入每个维度表示滑动窗口大小。必须是： :math:`[1, 1, k_d, k_h, k_w]` 或 :math:`[k_d, k_h, k_w]` 。如果 :math:`k_d = k_h = k_w` ，则可以输入整数。
        - **strides** (Union[int, tuple[int], list[int]]) - 长度为3或5的int列表。
          两个连续色块的中心在输入中的距离。必须是： :math:`[1, 1, s_d, s_h, s_w]` 或 :math:`[s_d, s_h, s_w]` 。如果 :math:`s_d = s_h = s_w` ，则可以输入整数。
        - **padding** (str) - 要使用的填充算法的类型。可选值有"SAME"和"VALID"。

    输入：
        - **input_x** (Tensor) - 一个五维的输入Tensor。数据类型必须为float16、float32，shape为 :math:`(x_n、x_c、x_d、x_h、x_w)` 。

    输出：
        Tensor，与输入的类型相同。如果填充为"VALID"，则shape为 :math:`(x_n, k_d * k_h * k_w * x_c, 1 + (x_d - k_d) / s_d,
        1 + (x_h - k_h) / s_h, 1 + (x_w - k_w) / s_w)` ；如果填充"SAME"，则shape为 :math:`(
        x_n, k_d * k_h * k_w * x_c, (x_d + s_d - 1) / s_d, (x_h + s_h - 1) / s_h, (x_w + s_w - 1) / s_w)` 。

    异常：
        - **TypeError** - 如果 `kernel_size` 或 `strides` 不是一个list，tuple或int。
        - **TypeError** - 如果 `input_x` 不是Tensor。
        - **TypeError** - 如果 `padding` 不是str。
        - **ValueError** - 如果 `kernel_size` 的长度不是3或5，并且 `kernel_size` 不是int。
        - **ValueError** - 如果 `strides` 的长度不是3或5，并且 `strides` 不是int。
        - **ValueError** - 如果 `padding` 既不是"VALID"也不是"SAME"。
        - **ValueError** - 如果 `kernel_size` 或 `strides` 的元素不是正整数。
        - **ValueError** - 如果 `input_x` 不是五维的Tensor。
        - **ValueError** - 如果 `input_x` 的shape含有0。
        - **ValueError** - 如果 `kernel_size` 或 `strides` 的前两个数不等于1。
        - **ValueError** - 如果 `padding` 为"VALID"，并且 :math:`input\_x - kernel\_size` 在d、h或w维上小于0。
        - **ValueError** - 如果 `padding` 为"SAME"，并且 :math:`padding\_needed = ((input\_x + strides - 1) / strides - 1) * strides + kernel\_size - input\_x` 在d、h或w维中小于0。
        - **ValueError** - 如果x_h不等于1或x_w不等于1，并且 :math:`x_w + padding\_needed - k_w - s_w` 小于0。
        - **ValueError** - 如果 :math:`x_d * x_h * x_w` 大于2048。
