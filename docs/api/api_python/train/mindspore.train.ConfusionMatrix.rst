mindspore.train.ConfusionMatrix
================================

.. py:class:: mindspore.train.ConfusionMatrix(num_classes, normalize='no_norm', threshold=0.5)

    计算混淆矩阵(confusion matrix)，通常用于评估分类模型的性能，包括二分类和多分类场景。

    如果只想使用混淆矩阵，请使用该类。如果想计算"PPV"、"TPR"、"TNR"等，请使用'mindspore.train.ConfusionMatrixMetric'类。

    参数：
        - **num_classes** (int) - 数据集中的类别数量。
        - **normalize** (str) - 计算ConfsMatrix的参数支持四种归一化模式，默认值：'no_norm'。

          - **"no_norm"** (None) - 不使用标准化。
          - **"target"** (str) - 基于目标值的标准化。
          - **"prediction"** (str) - 基于预测值的标准化。
          - **"all"** (str) - 整个矩阵的标准化。

        - **threshold** (float) - 阈值，用于与输入Tensor进行比较。默认值：0.5。

    .. py:method:: clear()

        重置评估结果。

    .. py:method:: eval()

        计算混淆矩阵。

        返回：
            numpy.ndarray，计算的结果。

    .. py:method:: update(*inputs)

        使用y_pred和y更新内部评估结果。

        参数：
            - ***inputs** (tuple) - 输入 `y_pred` 和 `y` 。 `y_pred` 和 `y` 是 `Tensor` 、列表或数组。
              `y_pred` 是预测值， `y` 是真实值， `y_pred` 的shape是 :math:`(N, C, ...)` 或 :math:`(N, ...)` ， `y` 的shape是 :math:`(N, ...)` 。

        异常：
            - **ValueError** - 输入参数的数量不等于2。
            - **ValueError** - 如果预测值和标签的维度不一致。
