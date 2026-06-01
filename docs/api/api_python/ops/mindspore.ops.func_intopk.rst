mindspore.ops.intopk
====================

.. py:function:: mindspore.ops.intopk(x1, x2, k)

    `mindspore.ops.intopk` 从2.9.0版本开始已被弃用，并将在未来版本中被移除。

    返回第二个输入tensor中的元素是否存在于第一个输入tensor的前 `k` 个元素中。

    参数：
        - **x1** (Tensor) - 二维输入tensor。
        - **x2** (Tensor) - 一维输入tensor。须满足 :math:`x2.shape[0] = x1.shape[0]`。
        - **k** (int) - 前 `k` 个元素。

    返回：
        一维的bool类型tensor，与 `x2` shape相同。
