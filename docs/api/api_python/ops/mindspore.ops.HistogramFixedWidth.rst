mindspore.ops.HistogramFixedWidth
=================================

.. py:class:: mindspore.ops.HistogramFixedWidth(nbins, dtype='int32')

    返回一个rank为1的直方图，该直方图中的每个组的值表示数量。每个组的宽度应该相等，且宽度由输入 `range` 和参数 `nbins` 决定。

    参数：
        - **nbins** (int) - 直方图的组数，类型为正整数。
        - **dtype** (str, optional) - 可选属性。数据类型必须为int32。默认值："int32"。

    输入：
        - **x** (Tensor) - HistogramFixedWidth的输入，为一个Tensor。数据类型必须为int32、float32或float16。
        - **range** (Tensor) - 数据类型与 `x` 相同，shape为 :math:`(2,)` 。x <= range[0] 将映射到histogram[0]，x >= range[1]将映射到histogram[-1]。

    输出：
        Tensor，数据类型为int32。

    异常：
        - **TypeError** - 如果 `dtype` 不是str或 `nbins` 不是int。
        - **ValueError** - 如果 `nbins` 小于1。
        - **ValueError** - 如果 `dtype` 既不是 `int32` 也不是 `int64` 。
