mindspore.train.HausdorffDistance
============================================

.. py:class:: mindspore.train.HausdorffDistance(distance_metric='euclidean', percentile=None, directed=False, crop=True)

    计算Hausdorff距离。Hausdorff距离是两个点集之间两点的最小距离的最大值，度量了两个点集间的最大不匹配程度。

    给定两个集合A和B，A和B之间的Hausdorff距离定义如下：

    .. math::
        H(A, B) = \text{max}[h(A, B), h(B, A)]
        h(A, B) = \underset{a \in A}{\text{max}}\{\underset{b \in B}{\text{min}} \rVert a - b \rVert \}
        h(B, A) = \underset{b \in B}{\text{max}}\{\underset{a \in A}{\text{min}} \rVert b - a \rVert \}

    其中 :math:`h(A, B)` 表示，对A中的每个点a找到B集合里的最近点，这些最短距离的最大值为从A到B的单向Hausdorff距离，同理，:math:`h(B, A)` 为集合B到集合A中最近点的最大距离。Hausdoff距离是有方向性的，通常情况下 :math:`h(A, B)` 不等于 :math:`h(B, A)`。:math:`H(A, B)` 为双向Hausdorff距离。

    参数：
        - **distance_metric** (string) - 支持如下三种距离计算方法："euclidean"、"chessboard" 或 "taxicab"。默认值："euclidean"。
        - **percentile** (float) - 0到100之间的浮点数。指定最终返回的Hausdorff距离的百分位数。默认值：None。
        - **directed** (bool) - 如果为True，为单向Hausdorff距离，只计算h(y_pred, y)距离；如果为False，为双向Hausdorff距离，计算max(h(y_pred, y), h(y, y_pred))。默认值：False。
        - **crop** (bool) - 是否裁剪输入图像，仅保留目标区域。为了保证y_pred和y的shape匹配，使用(y_pred | y)，即两图像的并集来确定bounding box。默认值：True。

    .. py:method:: clear()

        内部评估结果清零。

    .. py:method:: eval()

        计算定向或非定向Hausdorff距离。

        返回：
            numpy.float64，计算得到的Hausdorff距离。

        异常：
            - **RuntimeError** - 如果没有先调用update方法。

    .. py:method:: update(*inputs)

        使用 `y_pred`、`y` 和 `label_idx` 更新内部评估结果。

        参数：
            - **inputs** - `y_pred`、`y`  和 `label_idx`。`y_pred` 和 `y` 为Tensor， list或numpy.ndarray，`y_pred` 是预测的二值图像，`y` 是实际的二值图像。`label_idx` 的数据类型为int或float，表示像素点的类别值。

        异常：
            - **ValueError** - 输入的数量不等于3。
            - **TypeError** - label_idx 的数据类型不是int或float。
            - **ValueError** - label_idx 的值不在y_pred或y中。
            - **ValueError** - y_pred 和 y 的shape不同。
