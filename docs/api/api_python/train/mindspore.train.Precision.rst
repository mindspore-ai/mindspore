mindspore.train.Precision
==========================

.. py:class:: mindspore.train.Precision(eval_type='classification')

    计算数据分类的精度，包括单标签场景和多标签场景。

    此函数创建两个局部变量 :math:`\text{true_positive}` 和 :math:`\text{false_positive}`，用于计算精度。计算方式如下：

    .. math::
        \text{precision} = \frac{\text{true_positive}}{\text{true_positive} + \text{false_positive}}

    参数：
        - **eval_type** (str，可选) - 支持 ``'classification'`` 和 ``'multilabel'`` 。具体作用见下方的update方法。默认值： ``'classification'`` 。

    .. py:method:: clear()

        内部评估结果清零。

    .. py:method:: eval(average=False)

        计算精度。

        参数：
            - **average** (bool，可选) - 指定是否计算平均精度。默认值： ``False`` 。

        返回：
            numpy.float64，计算结果。

    .. py:method:: update(*inputs)

        使用预测值 :math:`y\_pred` 和真实标签 :math:`y` 更新局部变量。在多标签情况下， :math:`y` 和 :math:`y\_pred` 的元素必须为0或1。

        参数：
            - **inputs** - 输入 :math:`y\_pred` 和 :math:`y`。 :math:`y\_pred` 和 :math:`y` 支持Tensor、list或numpy.ndarray类型。

              - 对于 ``'classification'`` 情况， :math:`y\_pred` 在大多数情况下由范围 :math:`[0, 1]` 中的浮点数组成，shape为 :math:`(N, C)` ，其中 :math:`N` 是样本数， :math:`C` 是类别数。 :math:`y` 由整数值组成，如果是one_hot编码格式，shape是 :math:`(N, C)` ；如果是类别索引，shape是 :math:`(N,)` 。
              - 对于 ``'multilabel'`` 情况， :math:`y\_pred` 和 :math:`y` 只能是值为0或1的one-hot编码格式，其中值为1的索引表示正类别。 :math:`y\_pred` 和 :math:`y` 的shape都是 :math:`(N, C)` 。

        异常：
            - **ValueError** - inputs数量不是2。
