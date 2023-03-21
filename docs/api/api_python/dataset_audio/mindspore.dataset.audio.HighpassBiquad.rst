mindspore.dataset.audio.HighpassBiquad
======================================

.. py:class:: mindspore.dataset.audio.HighpassBiquad(sample_rate, cutoff_freq, Q=0.707)

    给音频波形上施加双二阶高通滤波器。

    接口实现方式类似于 `SoX库 <http://sox.sourceforge.net/sox.html>`_ 。

    参数：
        - **sample_rate** (int) - 波形的采样频率，如44100（单位：Hz），不能为零。
        - **cutoff_freq** (float) - 中心频率（单位：Hz）。
        - **Q** (float, 可选) - `品质因子 <https://zh.wikipedia.org/wiki/%E5%93%81%E8%B3%AA%E5%9B%A0%E5%AD%90>`_ ，取值范围为(0, 1]。默认值：0.707。

    异常：
        - **TypeError** - 当 `sample_rate` 的类型不为int。
        - **ValueError** - 当 `sample_rate` 的数值为0。
        - **TypeError** - 当 `cutoff_freq` 的类型不为float。
        - **TypeError** - 当 `Q` 的类型不为float。
        - **ValueError** - 当 `Q` 取值不在(0, 1]范围内。
        - **RuntimeError** - 当输入音频的shape不为<..., time>。
