mindspore.dataset.audio.AllpassBiquad
=====================================

.. py:class:: mindspore.dataset.audio.AllpassBiquad(sample_rate, central_freq, Q=0.707)

    给音频波形施加双极点全通滤波器，其中心频率和带宽由入参指定。

    全通滤波器能够改变音频频率与相位的关系，而不改变频率与幅度的关系。其系统函数为：

    .. math::
        H(s) = \frac{s^2 - \frac{s}{Q} + 1}{s^2 + \frac{s}{Q} + 1}

    接口实现方式类似于 `SoX库 <http://sox.sourceforge.net/sox.html>`_ 。
    
    .. note:: 待处理音频shape需为<..., time>。

    参数：
        - **sample_rate** (int) - 采样频率（单位：Hz），不能为零。
        - **central_freq** (float) - 中心频率（单位：Hz）。
        - **Q** (float, 可选) - `品质因子 <https://zh.wikipedia.org/wiki/%E5%93%81%E8%B3%AA%E5%9B%A0%E5%AD%90>`_ ，能够反映带宽与采样频率和中心频率的关系，取值范围为(0, 1]。默认值：0.707。

    异常：
        - **TypeError** - 当 `sample_rate` 的类型不为int。
        - **ValueError** - 当 `sample_rate` 的数值为0。
        - **TypeError** - 当 `central_freq` 的类型不为float。
        - **TypeError** - 当 `Q` 的类型不为float。
        - **ValueError** - 当 `Q` 取值不在(0, 1]范围内。
        - **RuntimeError** - 当输入音频的shape不为<..., time>。
