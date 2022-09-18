mindspore.train.F1
=====================

.. py:class:: mindspore.train.F1

    计算F1 score。F1是Fbeta的特殊情况，即beta为1。
    有关更多详细信息，请参阅类 :class:`mindspore.train.Fbeta`。

    .. math::
        F_1=\frac{2\cdot true\_positive}{2\cdot true\_positive + false\_negative + false\_positive}
