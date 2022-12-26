mindspore.ops.PSROIPooling
==========================

.. py:class:: mindspore.ops.PSROIPooling(spatial_scale, group_size, output_dim)

    对输入Tensor应用Position Sensitive ROI-Pooling。

    参数：
        - **spatial_scale** (float) - 将框坐标映射到输入坐标的比例因子。例如，如果你的框定义在224x224的图像上，并且你的输入是112x112的特征图（由原始图像的0.5倍缩放产生），此时需要将其设置为0.5。
        - **group_size** (int) - 执行池化后输出的大小（以像素为单位），以（高度，宽度）的格式输出。
        - **output_dim** (int) - 执行池化后输出的维度。

    输入：
        - **features** (Tensor) - 输入特征Tensor，其shape必须为 :math:`(N, C, H, W)` 。 各维度的值应满足： :math:`(C == output\_dim * group\_size * group\_size)` 。数据类型为float16或者float32。
        - **rois** (Tensor) - 其shape为 :math:`(batch, 5, rois_n)` ，数据类型为float16或者float32。第一个维度的batch为批处理大小。第二个维度的大小必须为5。第三维度rois_n是rois的数量。rois_n的值格式为：(index, x1, y1, x2, y2)。其中第一个元素是rois的索引。方框坐标格式为(x1、y1、x2、y2)，之后将把这些方框的选中的区域提取出来。区域坐标必须满足0 <= x1 < x2和0 <= y1 < y2。

    输出：
        - **out** (Tensor) - 池化后的输出。其shape为 :math:`(rois.shape[0] * rois.shape[2], output\_dim, group\_size, group\_size)` 。

    异常：
        - **TypeError** - `spatial_scale` 不是float类型。
        - **TypeError** - `group_size` 或者 `output_dim` 不是 int类型。
        - **TypeError** - `features` 或者 `rois` 不是Tensor。
        - **TypeError** - `rois` 数据类型不是float16或者float32。
        - **ValueError** - `features` 的shape不满足 :math:`(C == output\_dim * group\_size * group\_size)` 。
        - **ValueError** - `spatial_scale` 为负数。
