mindspore.ops.StridedSlice
===========================

.. py:class:: mindspore.ops.StridedSlice(begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0)

    对输入Tensor根据步长和索引进行切片提取。

    更多参考详见 :func:`mindspore.ops.strided_slice`。

    参数：
        - **begin_mask** (int，可选) - 表示切片的起始索引掩码。默认值： ``0`` 。
        - **end_mask** (int，可选) - 表示切片的结束索引掩码。默认值： ``0`` 。
        - **ellipsis_mask** (int，可选) - 维度掩码值为1说明不需要进行切片操作。为int型掩码。默认值： ``0`` 。
        - **new_axis_mask** (int，可选) - 表示切片的新增维度掩码。默认值： ``0`` 。
        - **shrink_axis_mask** (int，可选) - 表示切片的收缩维度掩码。为int型掩码。默认值： ``0`` 。

    输入：
        - **input_x** (Tensor) - 需要切片处理的输入Tensor。
        - **begin** (tuple[int]) - 指定开始切片的索引。
        - **end** (tuple[int]) - 指定结束切片的索引。
        - **strides** (tuple[int]) - 指定各维度切片的步长。输入为一个tuple，仅支持int值。`strides` 的元素必须非零。可能为负值，这会导致反向切片。

    输出：
        返回根据起始索引、结束索引和步长进行提取出的切片Tensor。
