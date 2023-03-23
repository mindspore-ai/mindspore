mindspore.dataset.audio.Contrast
================================

.. py:class:: mindspore.dataset.audio.Contrast(enhancement_amount=75.0)

    给音频波形施加对比度增强效果。

    与音频压缩相比，该效果通过修改音频信号使其听起来更响亮。

    接口实现方式类似于 `SoX库 <http://sox.sourceforge.net/sox.html>`_ 。

    .. note:: 待处理音频shape需为<..., time>。

    参数：
        - **enhancement_amount** (float, 可选) - 控制音频增益的量，取值范围为[0,100]。默认值：75.0。请注意当 `enhancement_amount` 等于0时，对比度增强效果仍然会很显著。

    异常：
        - **TypeError** - 当 `enhancement_amount` 的类型不为float。
        - **ValueError** - 当 `enhancement_amount` 取值不在[0, 100]范围内。
        - **RuntimeError** - 当输入音频的shape不为<..., time>。
