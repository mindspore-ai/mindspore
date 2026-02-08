mindspore.ops.RotatedIou
========================

.. py:class:: mindspore.ops.RotatedIou(trans=False, mode=0, is_cross=True, v_threshold=0.0, e_threshold=0.0)

    计算旋转矩形之间的重叠面积。

    .. note::
        Ascend平台支持的输入数据类型包括bfloat16、float16、float32。

    参数：
        - **trans** (bool，可选) - 区分 `boxes` 与 `query_boxes` 中矩形表示方法。如果为 ``True``，格式为 ``'xyxyt'``，如果为 ``False``，格式为 ``'xywht'``。默认为 ``False``。
        - **mode** (int，可选) - 区分计算模式。如果为 ``1``，计算方法为 ``'iof'``，如果为 ``0``，计算方法为 ``'iou'``。默认为 ``0``。
        - **is_cross** (bool，可选) - 如果为 ``True``，采用交叉计算，如果为 ``False``，表示一对一计算。默认为 ``True``。
        - **v_threshold** (float，可选) - 顶点判断的容忍阈值。默认为 ``0.0``。
        - **e_threshold** (float，可选) - 边相交判断的容忍阈值。默认为 ``0.0``。

    输入：
        - **boxes** (Tensor) - 第一组矩形，shape为 :math:`(B, N, 5)`。
        - **query_boxes** (Tensor) - 第二组矩形，shape为 :math:`(B, K, 5)`。

    输出：
        Tensor，shape为 :math:`(B, N, K)`。

    异常：
        - **TypeError** - `boxes` 不是Tensor。
        - **TypeError** - `query_boxes` 不是Tensor。
        - **ValueError** - `boxes` 与 `query_boxes` 第一个维度不相同。
        - **ValueError** - `boxes` 或 `query_boxes` 第三个维度不为 ``5``。