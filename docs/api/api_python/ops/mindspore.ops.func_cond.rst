mindspore.ops.cond
==================

.. py:function:: mindspore.ops.cond(A, p=None)

    返回给定Tensor的矩阵范数或向量范数。

    `p` 为norm的计算模式。支持下列norm模式。

    =================   ================================== ==============================================
    `p`                  矩阵范数                               向量范数
    =================   ================================== ==============================================
    ``None`` (默认值)     `2`-norm (参考最下方公式)            `2`-norm (参考最下方公式)
    ``'fro'``             Frobenius norm                     不支持
    ``'nuc'``             nuclear norm                       不支持
    ``inf``               :math:`max(sum(abs(x), dim=1))`    :math:`max(abs(x))`
    ``-inf``              :math:`min(sum(abs(x), dim=1))`    :math:`min(abs(x))`
    ``0``                 不支持                              :math:`sum(x != 0)`
    ``1``                 :math:`max(sum(abs(x), dim=0))`    参考最下方公式
    ``-1``                :math:`min(sum(abs(x), dim=0))`    参考最下方公式
    ``2``                 最大奇异值                           参考最下方公式
    ``-2``                最小奇异值                           参考最下方公式
    其余int或float值       不支持                              :math:`sum(abs(x)^{ord})^{(1 / ord)}`
    =================   ================================== ==============================================

    .. note::
        当前暂不支持复数。

    参数：
        - **A** (Tensor) - shape为 :math:`(*, n)` 或者 :math:`(*, m, n)` 的Tensor，其中*是零个或多个batch维度。
        - **p** (Union[int, float, inf, -inf, 'fro', 'nuc'], 可选) - norm的模式。行为参考上表。默认值： ``None`` 。

    返回：
        Tensor，进行条件数计算的结果，与输入 `A` 的数据类型相同。

    异常：
        - **TypeError** - `A` 是一个向量并且 `p` 是str类型。
        - **ValueError** - `A` 是一个矩阵并且 `p` 不是有效的取值。
        - **ValueError** - `A` 是一个矩阵并且 `p` 为一个取值不为[1, -1, 2, -2]之一的整型。
