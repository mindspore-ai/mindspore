mindspore.ops.AffineGrid
========================

.. py:class:: mindspore.ops.AffineGrid(align_corners=False)

    基于一批仿射矩阵 theta 生成一个2D 或 3D 的流场（采样网格）。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.affine_grid` 。

    参数：
        - **align_corners** (bool，可选) - 在几何上，我们将输入的像素视为正方形而不是点。如果设置为 ``True`` ，则极值 -1 和 1 指输入像素的中心。如果设置为 ``False`` ，则极值 -1 和 1 指输入像素的边角，从而使采样与分辨率无关。默认值： ``False`` 。

    输入：
        - **theta** (Tensor) - 仿射矩阵输入，其shape为 :math:`(N, 2, 3)` 用于 2D grid 或 :math:`(N, 3, 4)` 用于 3D grid。
        - **output_size** (tuple[int]) - 目标输出图像大小。指格式为 :math:`(N, C, H, W)` 的2D grid或 :math:`(N, C, D, H, W)` 的3D grid的大小。

    输出：
        Tensor，其数据类型与 `theta` 相同，其shape为 :math:`(N, H, W, 2)` 用于 2D grid或 :math:`(N, D, H, W, 3)` 用于 3D grid。
