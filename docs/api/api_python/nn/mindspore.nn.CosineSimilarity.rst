mindspore.nn.CosineSimilarity
=============================

.. py:class:: mindspore.nn.CosineSimilarity(similarity='cosine', reduction='none', zero_diagonal=True)

    计算表示相似性。

    **参数：** 

    - **similarity** (str) - "dot"或"cosine"。默认值："cosine"。
    - **reduction** (str) - "none"、"sum"或"mean"。默认值："none"。
    - **zero_diagonal** (bool) - 如果为True，则对角线将设置为零。默认值：True。

    .. py:method:: clear()

        重置评估结果。

    .. py:method:: eval()

        计算Cosine Similarity矩阵。

        **返回：**

        numpy.ndarray，相似度矩阵。

        **异常：**

        - **RuntimeError** - 如果没有先调用update方法。

    .. py:method:: update(*inputs)

        使用y_pred和y更新内部评估结果。

        **参数：** 

        - **inputs** (Union[Tensor, list, numpy.ndarray]) - 输入的矩阵。
