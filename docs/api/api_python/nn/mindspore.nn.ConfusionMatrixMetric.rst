mindspore.nn.ConfusionMatrixMetric
==================================

.. py:class:: mindspore.nn.ConfusionMatrixMetric(skip_channel=True, metric_name='sensitivity', calculation_method=False, decrease='mean')

    计算与混淆矩阵相关的度量。

    该计算基于全尺度张量，并收集批处理平均值，类通道数和迭代数。
    此函数支持计算参数metric_name中描述中列出的所有度量名称。
    
    如果要使用混淆矩阵计算，如"PPV"、"TPR"、"TNR"，请使用此类。
    如果只想计算混淆矩阵，请使用'mindspore.nn.ConfusionMatrix'。

    **参数：** 

    - **skip_channel** (bool) - 是否跳过预测输出的第一个通道的度量计算。默认值：True。
    - **metric_name** (str) - 建议采用如下指标。当然，也可以为这些指标设置通用别名。
      取值范围：["sensitivity", "specificity", "precision", "negative predictive value", "miss rate", "fall out", "false discovery rate", "false omission rate", "prevalence threshold", "threat score", "accuracy", "balanced accuracy", "f1 score", "matthews correlation coefficient", "fowlkes mallows index", "informedness", "markedness"]。
      默认值："sensitivity"。
    - **calculation_method** (bool) - 如果为True，则计算每个样本的度量值。如果为False，则累积所有样本的混淆矩阵。
      对于分类任务， `calculation_method` 应为False。默认值：False。
    - **decrease** (str) - 定义减少一批数据计算结果的模式。仅当 `calculation_method` 为True时，才生效。
      取值范围：["none", "mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel"]。默认值："mean"。

    .. py:method:: clear()

        重置评估结果。

    .. py:method:: eval()

        计算混淆矩阵度量。

        **返回：**

        numpy.ndarray，计算的结果。

    .. py:method:: update(*inputs)

        使用预测值和目标值更新状态。

        **参数：** 

        - **inputs** (tuple) - `y_pred` 和 `y` 。 `y_pred` 和 `y` 是 `Tensor` 、列表或数组。

          - **y_pred** (ndarray) - 待计算的输入数据。格式必须为one-hot，且第一个维度是batch。
            `y_pred` 的shape是 :math:`(N, C, ...)` 或 :math:`(N, ...)` 。
            至于分类任务， `y_pred` 的shape应为[BN]，其中N大于1。对于分割任务，shape应为[BNHW]或[BNHWD]。
          - **y** (ndarray) - 计算度量值的真实值。格式必须为one-hot，且第一个维度是batch。`y` 的shape是 :math:`(N, C, ...)` 。

        **异常：**

        - **ValueError** - 输入参数的数量不等于2。
