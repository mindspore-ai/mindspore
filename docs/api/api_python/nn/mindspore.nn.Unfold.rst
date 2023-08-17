mindspore.nn.Unfold
====================

.. py:class:: mindspore.nn.Unfold(ksizes, strides, rates, padding='valid')

    从图像中提取滑窗的区域块。
    
    输入为一个四维的Tensor，数据格式为(N, C, H, W)。

    参数：
        - **ksizes** (Union[tuple[int], list[int]]) - 滑窗大小，其格式为[1, ksize_row, ksize_col, 1]的int组成的tuple或list。
        - **strides** (Union[tuple[int], list[int]]) - 滑窗步长，其格式为[1, stride_row, stride_col, 1]的int组成的tuple或list。
        - **rates** (Union[tuple[int], list[int]]) - 滑窗元素之间的空洞个数，其格式为[1, rate_row, rate_col, 1] 的int组成的tuple或list。
        - **padding** (str) - 填充模式，可选值有： ``"same"`` 或 ``"valid"`` 的字符串，不区分大小写。默认值： ``"valid"`` 。

          - ``"same"`` - 指所提取的区域块的部分区域可以在原始图像之外，此部分填充为0。
          - ``"valid"`` - 表示所取的区域快必须被原始图像所覆盖。

    输入：
        - **x** (Tensor) - 输入四维Tensor，其shape为 :math:`[in\_batch, in\_depth, in\_row, in\_col]`，其数据类型为int。

    输出：
        Tensor，输出为四维Tensor，数据类型与 `x` 相同，其shape为 :math:`(out\_batch, out\_depth, out\_row, out\_col)`，且 `out_batch` 与 `in_batch` 相同。

        - :math:`out\_depth = ksize\_row * ksize\_col * in\_depth`
        - :math:`out\_row = (in\_row - (ksize\_row + (ksize\_row - 1) * (rate\_row - 1))) // stride\_row + 1`
        - :math:`out\_col = (in\_col - (ksize\_col + (ksize\_col - 1) * (rate\_col - 1))) // stride\_col + 1`

    异常：
        - **TypeError** - `ksize` ， `strides` 或 `rates` 既不是tuple，也不是list。
        - **ValueError** - `ksize` ， `strides` 或 `rates` 的shape不是 :math:`(1, x\_row, x\_col, 1)`。
        - **ValueError** - `ksize` ， `strides` 或 `rates` 的第二个和第三个元素小于1。