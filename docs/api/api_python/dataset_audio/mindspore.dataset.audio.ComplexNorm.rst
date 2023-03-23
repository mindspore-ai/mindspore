mindspore.dataset.audio.ComplexNorm
===================================

.. py:class:: mindspore.dataset.audio.ComplexNorm(power=1.0)

    计算复数序列的范数。

    .. note:: 待处理音频shape需为<..., complex=2>。第零维代表实部，第一维代表虚部。

    参数：
        - **power** (float, 可选) - 范数的幂，取值必须非负。默认值：1.0。

    异常：
        - **TypeError** - 当 `power` 的类型不为float。
        - **ValueError** - 当 `power` 为负数。
        - **RuntimeError** - 当输入音频的shape不为<..., complex=2>。
