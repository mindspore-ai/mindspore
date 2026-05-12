mindspore.mint.nn.functional.normalize
=========================================

.. py:function:: mindspore.mint.nn.functional.normalize(input, p=2.0, dim=1, eps=1e-12)

    将输入的张量按照指定维度进行归一化。

    对于一个输入的张量，其维度为 :math:`(n_{0},..., n_{dim},..., n_{k})`，对于第 :math:`n_{dim}` 个向量 `v`，它沿着维度 `dim` 按照如下公式进行转换：

    .. math::
        v=\frac{v}{\max(\left \| v \right \| _{p},\epsilon )}

    默认归一化计算方法为沿着第一个维度利用欧几里得范数进行归一化。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入的Tensor。
        - **p** (float) - 范数公式中的指数值。默认值为 ``2``。
        - **dim** (int) - 指定的维度。默认值为 ``1``。
        - **eps** (float) - 设置的最小值，以避免除法分母为 `0` 。默认值为 ``1e-12``。

    返回：
        Tensor。shape和数据类型与输入input相同。