mindspore.train.ROC
=====================

.. py:class:: mindspore.train.ROC(class_num=None, pos_label=None)

    计算ROC曲线。适用于求解二分类和多分类问题。在多分类的情况下，将基于one-vs-the-rest的方法进行计算。

    参数：
        - **class_num** (int) - 类别数。对于二分类问题，此入参可以不设置。默认值：None。
        - **pos_label** (int) - 正类的类别值。二分类问题中，不设置此入参，即 `pos_label` 为None时，正类类别值默认为1；用户可以自行设置正类类别值为其他值。多分类问题中，用户不应设置此参数，因为它将在[0,num_classes-1]范围内迭代更改。默认值：None。

    .. py:method:: clear()

        内部评估结果清零。

    .. py:method:: eval()

        计算ROC曲线。

        返回：
            tuple，由 `fpr`、`tpr` 和 `thresholds` 组成。

        - **fpr** (np.array) - 假正率。二分类情况下，返回不同阈值下的fpr；多分类情况下，则为fpr(false positive rate)的列表，列表的每个元素代表一个类别。
        - **tps** (np.array) - 真正率。二分类情况下，返回不同阈值下的tps；多分类情况下，则为tps(true positive rate)的列表，列表的每个元素代表一个类别。
        - **thresholds** (np.array) - 用于计算假正率和真正率的阈值。

        异常：
            - **RuntimeError** - 如果没有先调用update方法，则会报错。

    .. py:method:: update(*inputs)

        使用 `y_pred` 和 `y` 更新内部评估结果。

        参数：
            - **inputs** - 输入 `y_pred` 和 `y`。`y_pred` 和 `y` 是Tensor、list或numpy.ndarray。`y_pred` 一般情况下是范围为 :math:`[0, 1]` 的浮点数列表，shape为 :math:`(N, C)`，其中 :math:`N` 是用例数，:math:`C` 是类别数。`y` 为整数值，如果为one-hot格式，shape为 :math:`(N, C)`，如果是类别索引，shape为 :math:`(N,)`。
