mindspore.Tensor.pow
====================

.. py:method:: mindspore.Tensor.pow(power)

    计算Tensor中每个元素的 `power` 次幂。

    .. math::

        out_{i} = x_{i} ^{ y_{i}}

    .. note::
        - Tensor和 `power` 遵循 `隐式类型转换规则 <https://www.mindspore.cn/docs/zh-CN/r1.9/note/operator_list_implicit.html>`_ ，使数据类型保持一致。
        - 当前的Tensor和 `power` 的数据类型不能同时是bool，并保证其shape可以广播。

    参数：
        - **power** (Union[Tensor, number.Number, bool]) - 幂值，是一个number.Number或bool值，或数据类型为number或bool_的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为 `Tensor` 与 `power` 中精度较高的类型。

    异常：
        - **TypeError** - `power` 不是Tensor、number.Number或bool。
        - **ValueError** - 当Tensor和 `power` 都为Tensor时，它们的shape不相同。