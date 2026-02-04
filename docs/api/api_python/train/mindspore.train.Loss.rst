mindspore.train.Loss
====================

.. py:class:: mindspore.train.Loss

    计算loss的平均值。如果每 :math:`n` 次迭代调用一次 `update` 方法，则计算结果为：

    .. math::
        loss = \frac{\sum_{k=1}^{n}loss_k}{n}

    .. py:method:: clear()

        内部评估结果清零。

    .. py:method:: eval()

        计算loss的平均值。

        返回：
            float，loss的平均值。

        异常：
            - **RuntimeError** - 样本总数为0。

    .. py:method:: update(*inputs)

        更新内部评估结果。

        参数：
            - **inputs** - 输入只包含一个元素，且该元素为loss。loss的维度必须为0或1。

        异常：
            - **ValueError** - `inputs` 的长度不为1。
            - **ValueError** - loss的维度不为0或1。
