mindspore.train.Accuracy
=========================

.. py:class:: mindspore.train.Accuracy(eval_type='classification')

    计算数据分类的正确率，包括二分类和多分类。

    此类创建两个局部变量，预测正确的样本数和总样本数，用于计算 `y_pred` 和 `y` 的匹配率。此匹配率即为accuracy。

    .. math::
        \text{accuracy} =\frac{\text{true_positive} + \text{true_negative}}
        {\text{true_positive} + \text{true_negative} + \text{false_positive} + \text{false_negative}}

    参数：
        - **eval_type** (str) - 评估的数据集的类型，支持'classification'和'multilabel'。'classification'为单标签分类场景，'multilabel'为多标签分类场景。
          默认值：'classification'。

    .. py:method:: clear()

        内部评估结果清零。

    .. py:method:: eval()

        计算正确率。

        返回：
            np.float64，计算的正确率结果。

        异常：
            - **RuntimeError** - 样本量为0。

    .. py:method:: update(*inputs)

        更新局部变量。计算预测值y_pred和标签y的匹配频率。
        对于'classification'，如果预测的最大值的索引匹配真实的标签，预测正确；对于'multilabel'，如果预测值与真实标签匹配，预测正确。

        参数：
            - **inputs** - 预测值 `y_pred`  和真实标签 `y` ， `y_pred`  和 `y` 支持Tensor、list或numpy.ndarray类型。

              - 对于'classification'情况， `y_pred`  在大多数情况下由范围 :math:`[0, 1]` 中的浮点数组成，shape为 :math:`(N, C)` ，其中 :math:`N` 是样本数， :math:`C` 是类别数。
                `y` 由整数值组成，如果是one_hot编码格式，shape是 :math:`(N, C)` ；如果是类别索引，shape是 :math:`(N,)` 。
              - 对于'multilabel'情况， `y_pred` 和 `y` 只能是值为0或1的one-hot编码格式，其中值为1的索引表示正类别。 `y_pred` 和 `y` 的shape都是 :math:`(N, C)` 。

        异常：
            - **ValueError** - inputs的数量不等于2。
            - **ValueError** - 当前输入的 `y_pred` 和历史 `y_pred` 类别数不匹配。
