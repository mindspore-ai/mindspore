mindspore.train.Perplexity
===========================

.. py:class:: mindspore.train.Perplexity(ignore_label=None)

    计算困惑度（perplexity）。困惑度是衡量一个概率分布或语言模型好坏的标准。低困惑度表明语言模型可以很好地预测样本。计算方式如下：

    .. math::
        PP(W)=P(w_{1}w_{2}...w_{N})^{-\frac{1}{N}}=\sqrt[N]{\frac{1}{P(w_{1}w_{2}...w_{N})}}

    其中 :math:`w` 代表语料库中的单词.

    参数：
        - **ignore_label** (Union[int, None]) - 计数时要忽略的无效标签的索引。如果设置为None，它将包括所有条目。默认值：None。

    .. py:method:: clear()

        内部评估结果清零。

    .. py:method:: eval()

        返回当前评估结果。

        返回：
            numpy.float64，计算得到的困惑度结果。

        异常：
            - **RuntimeError** - 样本量为0。

    .. py:method:: update(*inputs)

        使用 `preds` 和 `labels` 更新内部评估结果。

        参数：
            - **inputs** - 输入 `preds` 和 `labels` 。 `preds` 和 `labels` 是Tensor、list或numpy.ndarray。 `preds` 是预测值， `labels` 是数据的标签。 `preds` 和 `labels` 的shape都是 :math:`(N, C)` 。

        异常：
            - **ValueError** - 输入数量不是2。
            - **RuntimeError** - 预测值和标签的长度不同。
            - **RuntimeError** - 预测值和标签的shape不同。
