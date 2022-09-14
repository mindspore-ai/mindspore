mindspore.nn.BleuScore
======================

.. py:class:: mindspore.nn.BleuScore(n_gram=4, smooth=False)

    计算BLEU分数。BLEU指的是具有一个或多个引用的机器翻译文本的metric。

    参数： 
        - **n_gram** (int) - 取值范围为1~4。默认值：4。
        - **smooth** (bool) - 是否采用平滑计算的方式。默认值：False。

    异常：
        - **ValueError** - `n_gram` 的取值范围不在1~4之间。

    .. py:method:: clear()

        重置评估结果。

    .. py:method:: eval()

        计算BLEU分数。

        返回：
            numpy.ndarray，numpy类型的BLEU分数。

        异常：
            - **RuntimeError** - 调用该方法前没有先调用update方法。

    .. py:method:: update(*inputs)

        使用输入的内容更新内部评估结果。

        参数： 
            - **inputs** (iterator) - 输入的元组，第一个输入是机器翻译语料库列表，第二个输入是引用语料库列表。

        异常：
            - **ValueError** - 输入参数的数量不等于2。
            - **ValueError** -  `candidate_corpus` 的长度与 `reference_corpus` 不同。