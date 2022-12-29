mindspore.ops.CombinedNonMaxSuppression
========================================

.. py:class:: mindspore.ops.CombinedNonMaxSuppression(clip_boxes=True, pad_per_class=False)

    根据分数降序，使用非极大抑制法通过遍历所有可能的边界框（Bounding Box）来选择一个最优的结果。

    参数：
        - **clip_boxes** (bool, 可选) - 如果为True，则假设边界框坐标在[0,1]之间，如果超出[0,1]则剪辑输出框。如果为False，则不进行剪切并按原样输出框坐标。默认值：True。
        - **pad_per_class** (bool, 可选) - 如果为True，输出 `nmsed_boxes` 、`nmsed_scores` 和 `nmsed_classes` 将被填充为 `max_output_size_per_class` * num_classes的长度，如果该长度超过 `max_total_size` ，在这种情况下它将被裁剪为 `max_total_size` 。如果为False，则输出 `nmsed_boxes` 、 `nmsed_scores` 和 `nmsed_classes` 将被填充/裁剪到 `max_total_size` 。默认值：False。

    输入：
        - **boxes** (Tensor) - Shape可表示为：(batch_size, num_boxes, q, 4)。如果q为1，则对所有类都使用相同的边界框；如果q等于类的数量，则对于每一类都使用特定的边界框。数据类型：float32。
        - **scores** (Tensor) - 表示对应于每个Bounding Boxes( `boxes` 的每一行)的单个分数，数据类型必须为float32，其shape可表示为：(batch_size, num_boxes, num_classes)。
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