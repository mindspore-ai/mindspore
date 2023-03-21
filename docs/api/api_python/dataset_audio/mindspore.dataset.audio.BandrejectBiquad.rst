mindspore.dataset.audio.BandrejectBiquad
========================================

.. py:class:: mindspore.dataset.audio.BandrejectBiquad(sample_rate, central_freq, Q=0.707)

    给音频波形施加双极点巴特沃斯（Butterworth）带阻滤波器。

    巴特沃斯滤波器的特点是通频带内的频率响应曲线最大限度平坦，没有纹波，而在阻频带则逐渐下降为零。

    巴特沃斯带阻滤波器的系统函数为：

    .. math::
        H(s) = \frac{s^2 + 1}{s^2 + \frac{s}{Q} + 1}

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
