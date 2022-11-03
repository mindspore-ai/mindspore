mindspore.dataset.audio.DeemphBiquad
====================================

.. py:class:: mindspore.dataset.audio.DeemphBiquad(sample_rate)

    给音频波形施加CD（IEC 60908）去重音（一种高音衰减搁置滤波器）效果。

    接口实现方式类似于 `SoX库 <http://sox.sourceforge.net/sox.html>`_ 。

    参数：
        - **sample_rate** (int) - 波形的采样频率，只能为44100或48000(Hz)。
    
    异常：
        - **TypeError** - 当 `sample_rate` 的类型不为int。
        - **ValueError** - 当 `sample_rate` 不为44100或48000。
        - **RuntimeError** - 当输入音频的shape不为<..., time>。
