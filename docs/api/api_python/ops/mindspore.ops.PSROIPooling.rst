mindspore.ops.PSROIPooling
==========================

.. py:class:: mindspore.ops.PSROIPooling(spatial_scale, group_size, output_dim)

    位置敏感的候选区域池化。

    参数：
        - **spatial_scale** (float) - 将标注框坐标映射到输入的比例因子。例如，如果标注框是在224x224图像的比例上定义的，并且您的输入是一个112x112的特征，那么您需要将其设置为0.5。
        - **group_size** (int) - 执行池化后输出的大小（以像素为单位），如（高度、宽度）。
        - **output_dim** (int) - 执行池化后输出的dim。

    输入：
        - **features** (Tensor) - 输入特征，其形状必须为 :math:`(N, C, H_{in}, W_{in})` ，数据类型为float16和float32。基于参数，此处应满足公式 :math:`(C == output_dim * group_size * group_size)` 。
        - **rois** (Tensor) - shape为 :math:`(batch, 5, rois_n)` 的Tensor，数据类型为float16和float32。第一维是批大小，第二维大小必须是5，第三维代表敏感区域的个数，其取值类似于(index, x1, y1, x2, y2)，index表示敏感区域序号，(x1，y1，x2，y2)表示敏感区域位置。

    输出：
        Tensor，其shape为 :math:`(rois.shape[0] * rois.shape[2], output_dim, group_size, group_size)` ，数据类型与 `features` 相同。
