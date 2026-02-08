mindspore.mint.cdist
=====================

.. py:function:: mindspore.mint.cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')

    计算两个Tensor每对行向量之间的p-norm距离。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        Ascend上支持的输入数据类型为float16和float32。

    参数：
        - **x1** (Tensor) - 输入Tensor，shape为 :math:`(B, P, M)` ， :math:`B` 表示0或者正整数。 :math:`B` 维度为0时该维度被忽略，shape为 :math:`(P, M)` 。
        - **x2** (Tensor) - 输入Tensor，shape为 :math:`(B, R, M)` ，与 `x1` 的数据类型一致。
        - **p** (float，可选) - 计算向量对p-norm距离的P值，P >= 0。默认值： ``2.0`` 。
        - **compute_mode** (str，可选) - 指定计算模式。目前设置此参数无效果。默认值： ``'use_mm_for_euclid_dist_if_necessary'`` 。

    返回：
        Tensor，p-范数距离，数据类型与 `x1` 一致，shape为 :math:`(B, P, R)`。

    异常：
        - **TypeError** - `x1` 或 `x2` 不是Tensor。
        - **TypeError** - `x1` 或 `x2` 的数据类型不符合上述“说明”中的要求。
        - **TypeError** - `p` 不是float32。
        - **ValueError** - `p` 是负数。
        - **ValueError** - `x1` 与 `x2` 维度不同。
        - **ValueError** - `x1` 与 `x2` 的维度既不是2，也不是3。
        - **ValueError** - `x1` 和 `x2` 的批次维无法广播。
        - **ValueError** - `x1` 和 `x2` 的列数不一样。
