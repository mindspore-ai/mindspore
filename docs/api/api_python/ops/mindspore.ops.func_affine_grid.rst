mindspore.ops.affine_grid
=========================

.. py:function:: mindspore.ops.affine_grid(theta, output_size, align_corners=False)

    基于一批仿射矩阵 `theta` ，返回一个 2D 或 3D 流场（采样网格）。

    参数：
        - **theta** (Tensor) - 仿射矩阵输入，其shape为 (N, 2, 3) 用于 2D 或 (N, 3, 4) 用于 3D grid。
        - **output_size** (tuple[int]) - 目标输出图像大小。其值为 (N, C, H, W) 用于 2D grid或 (N, C, D, H, W) 用于 3D grid。
        - **align_corners** (bool，可选) - 在几何上，我们将输入的像素视为正方形而不是点。如果设置为 ``True`` ，则极值 -1 和 1 被认为是指输入角像素的中心点。如果设置为 ``False`` ，则它们被认为是指输入角像素的角点，从而使采样与分辨率无关。默认值： ``False`` 。

    返回：
        Tensor，其数据类型与 `theta` 相同，其shape为 (N, H, W, 2) 用于 2D grid或 (N, D, H, W, 3) 用于 3D grid。

    异常：
        - **TypeError** - `theta` 不是Tensor或 `output_size` 不是tuple。
        - **ValueError** - `theta` 的shape不是 (N, 2, 3) 或 (N, 3, 4)。
        - **ValueError** - `output_size` 的长度不是 4 或 5。
        - **ValueError** - `theta` 的shape是 (N, 2, 3)，`output_size` 的长度却不是4； `theta` 的shape是 (N, 3, 4)，`output_size` 的长度却不是5。
        - **ValueError** - `output_size` 的第一个值不等于 `theta` 的第一维的长度。
