mindspore.ops.LeftShift
=======================

.. py:class:: mindspore.ops.LeftShift

    将Tensor每个位置的值向左移动若干个比特位。
    输入是两个Tensor，它们的数据类型必须一致，并且它们的shape可以广播。
    输出不支持隐式类型转换。

    .. math::

        \begin{aligned}
        &out_{i} =x_{i} << y_{i}
        \end{aligned}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **x1** (Tensor) - 目标Tensor，将根据 `x2` 对应位置的值向左移动相应的比特位，类型支持所有int和uint类型。
        - **x2** (Tensor) - Tensor必须具有与 `x1` 相同的数据类型，且其shape必须与 `x1` 相同或者可以与 `x1` 进行广播。

    输出：
        - **output** (Tensor) - 输出Tensor，数据类型与 `x1` 相同。并且输出Tensor的shape与 `x1` 相同，或者和 `x1` 和 `x2` 广播后的shape相同。

    异常：
        - **TypeError** - 如果 `x1` 或 `x2` 的数据类型错误。
        - **TypeError** - 如果 `x1` 或 `x2` 不是Tensor。
        - **ValueError** - 如果 `x1` 与 `x2` 无法广播。
