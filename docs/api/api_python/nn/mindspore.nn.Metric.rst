mindspore.nn.Metric
====================

.. py:class:: mindspore.nn.Metric

    用于计算评估指标的基类。

    在计算评估指标时需要调用 `clear` 、 `update` 和 `eval` 三个方法，在继承该类自定义评估指标时，也需要实现这三个方法。其中，`update` 用于计算中间过程的内部结果，`eval` 用于计算最终评估结果，`clear` 用于重置中间结果。
    请勿直接使用该类，需使用子类如 :class:`mindspore.nn.MAE` 、 :class:`mindspore.nn.Recall` 等。

    .. py:method:: clear()
        :abstractmethod:

        描述了清除内部评估结果的行为。

        .. note::
            所有子类都必须重写此接口。

    .. py:method:: eval()
        :abstractmethod:

        描述了计算最终评估结果的行为。

        .. note::
            所有子类都必须重写此接口。

    .. py:method:: indexes
        :property:

        获取当前的 `indexes` 值。默认为None，调用 `set_indexes` 可修改 `indexes` 值。

    .. py:method:: set_indexes(indexes)

        该接口用于重排 `update` 的输入。

        给定(label0, label1, logits)作为 `update` 的输入，将 `indexes` 设置为[2, 1]，则最终使用(logits, label1)作为 `update` 的真实输入。

        .. note::
            在继承该类自定义评估函数时，需要用装饰器 `mindspore.nn.rearrange_inputs` 修饰 `update` 方法，否则配置的 `indexes` 值不生效。


        **参数：**

        - **indexes** (List(int)) - logits和标签的目标顺序。

        **输出：**

        :class:`Metric` ，类实例本身。

        **样例：**

        >>> import numpy as np
        >>> from mindspore import nn, Tensor
        >>>
        >>> x = Tensor(np.array([[0.2, 0.5], [0.3, 0.1], [0.9, 0.6]]))
        >>> y = Tensor(np.array([1, 0, 1]))
        >>> y2 = Tensor(np.array([0, 0, 1]))
        >>> metric = nn.Accuracy('classification').set_indexes([0, 2])
        >>> metric.clear()
        >>> # indexes为[0, 2]，使用x作为预测值，y2作为真实标签
        >>> metric.update(x, y, y2)
        >>> accuracy = metric.eval()
        >>> print(accuracy)
        0.3333333333333333

    .. py:method:: update(*inputs)
        :abstractmethod:

        描述了更新内部评估结果的行为。

        .. note::
            所有子类都必须重写此接口。

        **参数：**

        - **inputs** - 可变长度输入参数列表。通常是预测值和对应的真实标签。