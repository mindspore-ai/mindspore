mindspore.train.MeanSurfaceDistance
===============================================

.. py:class:: mindspore.train.MeanSurfaceDistance(symmetric=False, distance_metric='euclidean')

    计算从 `y_pred` 到 `y` 的平均表面距离。通常情况下，用来衡量分割任务中，预测情况和真实情况之间的差异度。

    给定两个集合A和B，S(A)表示A的表面像素，任意v到S(A)的最短距离定义为：

    .. math::
        {\text{dis}}\left (v, S(A)\right ) = \underset{s_{A}  \in S(A)}{\text{min }}\rVert v - s_{A} \rVert

    从集合B到集合A的平均表面距离(Average Surface Distance)为：

    .. math::
        AvgSurDis(B \rightarrow A) = \frac{\sum_{s_{B}  \in S(B)}^{} {\text{dis}  \left
        ( s_{B}, S(A) \right )} } {\left | S(B) \right |}

    其中 \|\|\*\|\| 表示距离度量。 \|\*\| 表示元素的数量。

    从集合B到集合A以及从集合A到集合B的表面距离平均值为：

    .. math::
        MeanSurDis(A \leftrightarrow B) = \frac{\sum_{s_{A}  \in S(A)}^{} {\text{dis}  \left ( s_{A}, S(B) \right )}
        + \sum_{s_{B}  \in S(B)}^{} {\text{dis}  \left ( s_{B}, S(A) \right )} }{\left | S(A) \right | +
        \left | S(B) \right |}

    参数：
        - **distance_metric** (string) - 支持如下三种距离计算方法："euclidean"、"chessboard"或"taxicab"。默认值："euclidean"。
        - **symmetric** (bool) - 是否计算 `y_pred` 和 `y` 之间的对称平均平面距离。如果为False，计算方式为  :math:`AvgSurDis(y\_pred \rightarrow y)` , 如果为True，计算方式为 :math:`MeanSurDis(y\_pred \leftrightarrow y)` 。默认值：False。

    .. py:method:: clear()

       内部评估结果清零。

    .. py:method:: eval()

        计算平均表面距离。

        返回：
            numpy.float64，计算得到的平均表面距离值。

        异常：
            - **RuntimeError** - 如果没有先调用update方法。

    .. py:method:: update(*inputs)

        使用 `y_pred`、`y` 和 `label_idx` 更新内部评估结果。

        参数：
            - **inputs** - `y_pred`、`y` 和 `label_idx`。`y_pred` 和 `y` 为Tensor，list或numpy.ndarray，`y_pred` 是预测的二值图像。`y` 是实际的二值图像。`label_idx` 数据类型为int或float，表示像素点的类别值。

        异常：
            - **ValueError** - 输入的数量不等于3。
            - **TypeError** - `label_idx` 的数据类型不是int或float。
            - **ValueError** - `label_idx` 的值不在y_pred或y中。
            - **ValueError** - `y_pred` 和 `y` 的shape不同。
