mindspore.dataset.audio.ComputeDeltas
=====================================

.. py:class:: mindspore.dataset.audio.ComputeDeltas(win_length=5, pad_mode=BorderType.EDGE)

    计算频谱的delta系数，也叫差分系数。

    delta系数能够帮助理解功率谱中的动态信息，可通过下列公式进行计算：

    .. math::
        d_{t}=\frac{{\textstyle\sum_{n=1}^{N}}n(c_{t+n}-c_{t-n})}{2{\textstyle\sum_{n=1}^{N}}n^{2}}

    其中， :math:`d_{t}` 是 :math:`t` 时刻的delta值， :math:`c_{t}` 是 :math:`t` 时刻的频谱系数， :math:`N` 是 :math:`(\text{win_length} - 1) // 2` 。

    参数：
        - **win_length** (int, 可选) - 用于计算delta值的窗口长度，必须不小于3。默认值： ``5`` 。
        - **pad_mode** (:class:`~.audio.BorderType`, 可选) - 边界填充模式，可为
          ``BorderType.CONSTANT`` 、 ``BorderType.EDGE`` 、 ``BorderType.REFLECT`` 或 ``BorderType.SYMMETRIC`` 。
          默认值： ``BorderType.EDGE`` 。

          - ``BorderType.CONSTANT`` ，使用常量值填充。
          - ``BorderType.EDGE`` ，使用各边的边界像素值填充。
          - ``BorderType.REFLECT`` ，以各边的边界为轴进行镜像填充，忽略边界像素值。
            例如，向 [1, 2, 3, 4] 的两边分别填充2个元素，结果为 [3, 2, 1, 2, 3, 4, 3, 2]。
          - ``BorderType.SYMMETRIC`` ，以各边的边界为轴进行对称填充，包括边界像素值。
            例如，向 [1, 2, 3, 4] 的两边分别填充2个元素，结果为 [2, 1, 1, 2, 3, 4, 4, 3]。

    异常：
        - **TypeError** - 当 `win_length` 的类型不为int。
        - **ValueError** - 当 `win_length` 小于3。
        - **TypeError** - 当 `pad_mode` 的类型不为 :class:`mindspore.dataset.audio.BorderType` 。
        - **RuntimeError** - 当输入音频的shape不为<..., freq, time>。

    教程样例：
        - `音频变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/audio_gallery.html>`_
