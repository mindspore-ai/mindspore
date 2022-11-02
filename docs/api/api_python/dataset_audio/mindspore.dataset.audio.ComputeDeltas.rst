mindspore.dataset.audio.ComputeDeltas
=====================================

.. py:class:: mindspore.dataset.audio.ComputeDeltas(win_length=5, pad_mode=BorderType.EDGE)

    计算光谱图的增量系数。

    .. math::
        d_{t}=\frac{{\textstyle\sum_{n=1}^{N}}n(c_{t+n}-c_{t-n})}{2{\textstyle\sum_{n=1}^{N}}n^{2}}

    其中， :math:`d_{t}` 是时间 :math:`t` 的增量， :math:`c_{t}` 是时间 :math:`t` 的频谱图系数， :math:`N` 是 :math:`(\text{win_length}-1)//2` 。

    参数：
        - **win_length** (int, 可选) - 计算窗口长度，长度必须不小于3，默认值：5。
        - **pad_mode** (:class:`mindspore.dataset.audio.BorderType`, 可选) - 边界填充模式，可以是
          [BorderType.CONSTANT, BorderType.EDGE, BorderType.REFLECT, BordBorderTypeer.SYMMETRIC]中任何一个。
          默认值：BorderType.EDGE。

          - BorderType.CONSTANT，用常量值填充边界。
          - BorderType.EDGE，使用各边的边界像素值进行填充。
          - BorderType.REFLECT，以各边的边界为轴进行镜像填充，忽略边界像素值。例如，对 [1,2,3,4] 的两侧分别填充2个元素，结果为 [3,2,1,2,3,4,3,2]。
          - BorderType.SYMMETRIC，以各边的边界为轴进行对称填充，包括边界像素值。例如，对 [1,2,3,4] 的两侧分别填充2个元素，结果为 [2,1,1,2,3,4,4,3]。
