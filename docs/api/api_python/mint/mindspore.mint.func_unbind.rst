mindspore.mint.unbind
=====================

.. py:function:: mindspore.mint.unbind(input, dim=0)

    沿指定轴分解Tensor。给定一个shape为 :math:`(n_1, n_2, ..., n_R)` 的Tensor，沿指定的 `dim` 对其分解，
    返回多个shape为 :math:`(n_1, n_2, ..., n_{dim}, n_{dim+2}, ..., n_R)` 的Tensor。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 用来分解的输入Tensor。其shape为 :math:`(n_1, n_2, ..., n_R)` 。其rank必须大于0。
        - **dim** (int，可选) - 用以分解的维度。取值范围为[-R, R)。默认值为 ``0`` 。

    返回：
        Tensor组成的tuple。每个Tensor对象的shape相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `dim` 不是int类型。
        - **ValueError** - 如果 `dim` 超出[-R, R)范围。