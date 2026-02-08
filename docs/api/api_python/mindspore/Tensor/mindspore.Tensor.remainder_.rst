mindspore.Tensor.remainder\_
============================

.. py:method:: mindspore.Tensor.remainder_(other) -> Tensor

    逐元素计算 `self` 除以 `other` 后的余数。结果与除数 `other` 同号且绝对值小于除数的绝对值。

    .. code-block::

        remainder(self, other) == self - self.div(other, rounding_mode="floor") * other

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        - 输入不支持复数类型。
        - 被除数 `self` 是数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 的Tensor。
        - 当 `self` 和 `other` 具有不同的shape时， `other` 必须能向 `self` 广播。

    参数：
        - **other** (Union[Tensor, number, bool]) - 除数为数值型、bool，或者数据类型为数值型或bool的Tensor。

    返回：
        Tensor，shape与 `self` 的shape相同，数据类型和 `self` 的数据类型相同。

    异常：
        - **RuntimeError** - 如果 `other` 不能向 `self` 广播。
