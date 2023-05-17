mindspore.ops.LogSpace
======================

.. py:class:: mindspore.ops.LogSpace(steps=10, base=10, dtype=mstype.float32)

    返回一个大小为 `steps` 的1-D Tensor，其值从 :math:`base^{start}` 到 :math:`base^{end}` ，以 `base` 为底数。

    .. math::
        \begin{aligned}
        &step = (end - start)/(steps - 1)\\
        &output = [base^{start}, base^{start + 1 * step}, ... , base^{start + (steps-2) * step}, base^{end}]
        \end{aligned}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **steps** (int，可选) - `steps` 必须为非负整数。默认值： ``10`` 。
        - **base** (int，可选) - `base` 必须为非负整数。默认值： ``10`` 。
        - **dtype** (mindspore.dtype，可选) - 输出的数据类型，支持 ``mstype.float16`` 、 ``mstype.float32`` 或 ``mstype.float64`` 。默认值： ``mstype.float32`` 。

    输入：
        - **start** (Tensor) - 间隔的起始值，shape为0-D，数据类型为float16、float32或float64（对于GPU）类型。
        - **end** (Tensor) - 间隔的结束值，shape为0-D，数据类型为float16、float32或float64（对于GPU）类型。

    输出：
        Tensor，shape为 :math:`(step, )` ，数据类型由属性 `dtype` 设置。

    异常：
        - **TypeError** - 若 `input` 不是一个Tensor。
        - **TypeError** - 若 `steps` 不是一个整数。
        - **TypeError** - 若 `base` 不是一个整数。
        - **TypeError** - 若 `dtype` 不是mindspore.float16、mindspore.float32或mindspore.float64。
        - **ValueError** - 若 `steps` 不是非负整数。
        - **ValueError** - 若 `base` 不是非负整数。
