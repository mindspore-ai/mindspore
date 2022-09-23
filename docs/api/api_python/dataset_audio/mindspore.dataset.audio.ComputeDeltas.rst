mindspore.dataset.audio.ComputeDeltas
=====================================

.. py:class:: mindspore.dataset.audio.ComputeDeltas(win_length=5, pad_mode=BorderType.EDGE)

    计算光谱图的增量系数。

    .. math::
        d_{t}=\frac{{\textstyle\sum_{n=1}^{N}}n(c_{t+n}-c_{t-n})}{2{\textstyle\sum_{n=1}^{N}}n^{2}}

    参数：
        - **win_length** (int, 可选) - 计算窗口长度，长度必须不小于3，默认值：5。
        - **pad_mode** (:class:`mindspore.dataset.audio.BorderType`, 可选) - 边界填充模式，默认值：BorderType.EDGE。
