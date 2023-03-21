mindspore.dataset.audio.Angle
=============================

.. py:class:: mindspore.dataset.audio.Angle

    计算复数序列的角度。

    .. note:: 待处理音频shape需为<..., complex=2>。第零维代表实部，第一维代表虚部。

    异常：
        - **RuntimeError** - 当输入音频的shape不为<..., complex=2>。
