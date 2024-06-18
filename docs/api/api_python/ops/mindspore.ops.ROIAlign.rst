mindspore.ops.ROIAlign
========================

.. py:class:: mindspore.ops.ROIAlign(pooled_height, pooled_width, spatial_scale, sample_num=2, roi_end_mode=1)

    感兴趣区域对齐(RoI Align)运算。

    RoI Align通过在特征图上对附近网格点进行双线性插值计算每个采样点。RoI Align不对RoI、其子区域或采样点的中任何坐标执行量化。参阅论文 `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_ 。

    参数：
        - **pooled_height** (int) - 输出特征高度。
        - **pooled_width** (int) - 输出特征宽度。
        - **spatial_scale** (float) - 缩放系数，将原始图像坐标映射到输入特征图坐标。
          设RoI的高度在原始图像中为 `ori_h` ，在输入特征图中为 `fea_h` ，则 `spatial_scale` 应为 `fea_h / ori_h` 。
        - **sample_num** (int) - 采样数。默认值： ``2`` 。
        - **roi_end_mode** (int) - 值必须为0或1。
          如果值为0，则使用该算子的历史实现。
          如果值为1，则对RoI末尾的像素进行偏移，偏移量为 `+1*spatial_scale` 。
          默认值： ``1`` 。

    输入：
        - **features** (Tensor) - 输入特征，shape: :math:`(N, C, H, W)` 。数据类型支持float16和float32。
        - **rois** (Tensor) - shape: :math:`(rois\_n, 5)` 。数据类型支持float16和float32。
          `rois_n` 为RoI的数量。第二个维度的大小必须为 `5` ，分别代表 :math:`(image\_index, top\_left\_x, top\_left\_y, bottom\_right\_x, bottom\_right\_y)` 。
          `image_index` 表示图像的索引； `top_left_x` 和 `top_left_y` 分别对应RoI左上角坐标的 `x` 和 `y` 值； 
          `bottom_right_x` 和 `bottom_right_y` 分别对应RoI右下角坐标的 `x` 和 `y` 值。

    输出：
        Tensor，shape: :math:`(rois\_n, C, pooled\_height, pooled\_width)` 。

    异常：
        - **TypeError** - `pooled_height` 、`pooled_width` 、`sample_num` 或 `roi_end_mode` 不是int类型。
        - **TypeError** - `spatial_scale` 不是float类型。
        - **TypeError** - `features` 或 `rois` 不是Tensor。
