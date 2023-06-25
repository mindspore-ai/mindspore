mindspore.ops.MatrixBandPart
============================

.. py:class:: mindspore.ops.MatrixBandPart

    提取一个Tensor中每个矩阵的中心带，中心带之外的所有值都设置为零。

    更多参考详见 :func:`mindspore.ops.matrix_band_part` 。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **x** (Tensor) - `x` 的shape为 :math:`(*, m, n)` ，其中 :math:`*` 表示任意batch维度。
        - **lower** (Union[int, Tensor]) - 要保留的下部子对角线数。其数据类型必须是int32或int64。如果为负数，则保留整个下三角形。
        - **upper** (Union[int, Tensor]) - 要保留的上部子对角线数。其数据类型必须是int32或int64。如果为负数，则保留整个上三角形。

    输出：
        Tensor，其数据类型和维度必须和输入中的 `x` 保持一致。
