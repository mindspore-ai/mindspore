mindspore.ops.affine_grid
=========================

.. py:function:: mindspore.ops.affine_grid(theta, size, align_corners=False)

    基于输入的批量仿射矩阵 `theta` ，返回一个二维或三维的流场（采样网格）。

    参数：
        - **theta** (Tensor) - 仿射矩阵输入，其shape为 :math:`(N, 2, 3)` 用于 2D grid 或 :math:`(N, 3, 4)` 用于 3D grid。
        - **size** (tuple[int]) - 目标输出图像大小。指格式为 :math:`(N, C, H, W)` 的2D grid或 :math:`(N, C, D, H, W)` 的3D grid的大小。
        - **align_corners** (bool，可选) - 在几何上，我们将输入的像素视为正方形而不是点。如果设置为 ``True`` ，则极值 -1 和 1 指输入像素的中心。如果设置为 ``False`` ，则极值 -1 和 1 指输入像素的边角，从而使采样与分辨率无关。默认值： ``False`` 。

    返回：
        Tensor，其数据类型与 `theta` 相同，其shape为 :math:`(N, H, W, 2)` 用于 2D grid或 :math:`(N, D, H, W, 3)` 用于 3D grid。

    异常：
        - **TypeError** - `theta` 不是Tensor或 `size` 不是tuple。
        - **ValueError** - `theta` 的shape不是 :math:`(N, 2, 3)` 或 :math:`(N, 3, 4)` 。
        - **ValueError** - `size` 的长度不是 4 或 5。
        - **ValueError** - `theta` 的shape是 :math:`(N, 2, 3)` ，`size` 的长度却不是4； `theta` 的shape是 :math:`(N, 3, 4)` ，`size` 的长度却不是5。
        - **ValueError** - `size` 的第一个值不等于 `theta` 的第一维的长度。
