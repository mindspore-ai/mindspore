mindspore.Tensor.sub
====================

.. py:method:: mindspore.Tensor.sub(other, *, alpha=1) -> Tensor

    对 `other` 缩放 `alpha` 后与 `self` 相减。

    .. math::

        out_{i} = self_{i} - alpha \times other_{i}

    .. note::
        - 当两个输入具有不同的shape时，它们的shape必须要能广播为一个共同的shape。
        - 两个输入和 `alpha` 遵循隐式类型转换规则，使数据类型保持一致。

    参数：
        - **other** (Union[Tensor, number.Number, bool]) - 另外一个输入，数据类型为 `number.Number`、 `bool` 或者数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html>`_ 或 `bool <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html>`_ 的Tensor。

    关键字参数：
        - **alpha** (number.Number，可选) - 应用于 `other` 的缩放因子，默认值为 ``1`` 。

    返回：
        Tensor。其shape与 `self` 、 `other` 广播后的shape相同，并且数据类型是两个输入和 `alpha` 之间具有更高精度，或位数更多的类型。

    异常：
        - **TypeError** - 如果 `other` 或者 `alpha` 的类型不是以下类型： `number.Number`、 `bool` 或 `Tensor`。
        - **TypeError** - 如果 `alpha` 是浮点类型，但是 `self` 和 `other` 却不是浮点类型。
        - **TypeError** - 如果 `alpha` 是bool类型，但是 `self` 和 `other` 却不是bool类型。

    .. py:method:: mindspore.Tensor.sub(y) -> Tensor
        :noindex:

    详情请参考 :func:`mindspore.ops.sub` 。
