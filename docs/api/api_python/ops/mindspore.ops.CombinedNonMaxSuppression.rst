mindspore.ops.CombinedNonMaxSuppression
========================================

.. py:class:: mindspore.ops.CombinedNonMaxSuppression(clip_boxes=True, pad_per_class=False)

    使用非极大值抑制法遍历候选边界框列表，从中选择一组子集，其中边界框按其置信度得分降序排列。

    参数：
        - **clip_boxes** (bool, 可选) - 确定是否应用边界框归一化，以确保坐标在[0,1]范围内。
          默认值：True。

          - 如果为True，则剪裁超出此范围的框。
          - 如果为False，则返回框坐标，而不进行任何修改。

        - **pad_per_class** (bool, 可选) - 确定是否需要对非极大值抑制（NMS）算法的输出进行填充或剪裁，以满足最大尺寸的限制。
          默认值：False。

          - 如果为False，则将输出剪裁到最大尺寸 `max_total_size` 。
          - 如果为True，则将输出填充到 `max_size_per_class` * `num_classes` 的最大长度，如果超 `过max_total_size` 则剪裁。

    输入：
        - **boxes** (Tensor) - 边界框坐标，是一个float32类型的Tensor，shape为 :math:`(batch_size, num_boxes, q, 4)` 。如果q为1，则所有类别都使用相同的边界框。否则，如果q等于类的数量，则对于每一类都使用特定的边界框。
        - **scores** (Tensor) - 表示对应于每个Bounding Boxes( `boxes` 的每一行)的单个分数，数据类型必须为float32，其shape可表示为： :math:`(batch_size, num_boxes, num_classes)` 。
        - **max_output_size_per_class** (Tensor) - 0D Tensor，表示每个类中由非极大抑制法（non-max suppression）选择的Bounding Boxes数目的上限。数据类型：int32。
        - **max_total_size** (Tensor) - 0D Tensor，表示在所有类中可保留的Bounding Boxes数目的上限。数据类型：int32。
        - **iou_threshold** (Tensor) - 0D Tensor，判断Bounding Boxes是否与IOU重叠过多的阈值，取值必须在[0,1]区间内。数据类型：float32。
        - **score_threshold** (Tensor) - 0D Tensor，表示根据 `score` 判断何时删除Bounding Boxes的阈值。数据类型：float32。

    输出：
        - **nmsed_boxes** (Tensor) - 包含由非极大抑制法选择出来的Bounding Boxes，shape为(batch_size, num_detection, 4)，数据类型为float32。
        - **nmsed_scores** (Tensor) - 包含每个box的分数，shape为(batch_size, num_detection)，数据类型为float32。
        - **nmsed_classes** (Tensor) - 包含每个box的类别，shape为(batch_size, num_detection)，数据类型为float32。
        - **valid_detections** (Tensor) - 表示每个batch的有效检测数，shape为(batch_size,)，数据类型为int32。

    异常：
        - **TypeError** -  `boxes` 、 `scores` 、 `iou_threshold` 、 `score threshold` 的数据格式不是float32。
        - **TypeError** -  `max_output_size_per_class` 和 `max_total_size` 的数据格式不是int32。
        - **ValueError** -  `boxes` 不是四维Tensor。
        - **ValueError** -  `max_output_size_per_class` 、 `max_total_size` 、 `iou_threshold` 和 `score threshold` 不是0D Tensor。
        - **ValueError** -  `boxes` 和 `scores` 的shape[0]或shape[1]不一致。
        - **ValueError** -  `boxes` 和 `scores` 的shape[2]不一致或 `boxes` 的shape[2]不为1。
        - **ValueError** -  `scores` 不是3D Tensor。
        - **ValueError** -  `max_total_size` 小于0。
        - **ValueError** -  `max_output_size_per_class` 小于0。
        - **ValueError** -  `iou_threshold` 取值不在区间[0,1]中。
