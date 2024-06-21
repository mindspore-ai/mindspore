mindspore.mint.arange
=====================

.. py:function:: mindspore.mint.arange(start=0, end=None, step=1, *, dtype=None)

    创建一个从 `start` 开始， `end` 结束（不含），步长为 `step` 序列（一维张量）。

    参数：
        - **start** (Union[float, int]，可选) - 序列起始值，默认值： ``0`` 。
        - **end** (Union[float, int]，可选) - 序列终止值（不含），默认值： ``None`` ，如果是 ``None`` ，则将 `start` 作为终止值， ``0`` 为起始值。
        - **step** (Union[float, int]，可选) - 序列步长，默认值： ``1`` 。

    关键字参数：
        - **dtype** (mindspore.dtype，可选) - 输出Tensor的dtype。默认值： ``None`` 。

          当该参数未设置或为 ``None`` 时：

          如果 `start` 、 `end` 和 `step` 都是int，则输出Tensor的dtype为int64。

          如果 `start` 、 `end` 和 `step` 中包含float，则输出Tensor的dtype为float32。

    返回：
        一维张量，如果设置了`dtype`参数，则会被cast成该类型的Tensor，有可能因此损失精度。

    异常：
        - **TypeError** - `start` ， `end` 和 `step` 不是int或float类型。
        - **ValueError** - `step` = 0。
        - **ValueError** - `step` > 0 且 `start` >= `end`。
        - **ValueError** - `step` < 0 且 `start` <= `end`。
