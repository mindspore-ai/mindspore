mindspore.mint.sub
===========================

.. py:function:: mindspore.mint.sub(input, other, *, alpha=1) -> Tensor

    对 `other` 缩放 `scalar` 后与 `input` 相减。

    .. math::

        out_{i} = input_{i} - alpha \times other_{i}

    .. note::
        - 当两个输入shape不同时，
          它们必须能够广播到一个共同的shape。
        - 两个输入和 `alpha` 遵守隐式类型转换规则以使数据类型保持一致。

    参数：
        - **input** (Union[Tensor, number.Number, bool]) - 第一个输入是number.Number、
          bool或数据类型为
          `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_ 或
          `bool <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_ 的Tensor。
        - **other** (Union[Tensor, number.Number, bool]) - 第二个输入，是number.Number、bool或数据类型为
          `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_ 或
          `bool <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_ 的Tensor。

    关键字参数：
        - **alpha** (number.Number，可选) - 应用于 `other` 的缩放因子，默认值为 ``1``。

    返回：
        Tensor，其shape与输入 `input`、 `other` 广播后的shape相同，
        数据类型是两个输入和 alpha 中精度更高的类型。

    异常：
        - **TypeError** - 如果 `input`、 `other` 不是以下类型之一：Tensor、number.Number、bool。
        - **TypeError** - 如果 `alpha` 是 float 类型，但是 `input`、 `other` 不是 float 类型。
        - **TypeError** - 如果 `alpha` 是 bool 类型，但是 `input`、 `other` 不是 bool 类型。

