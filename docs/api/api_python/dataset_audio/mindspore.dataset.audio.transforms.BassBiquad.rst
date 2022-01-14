mindspore.dataset.audio.transforms.BassBiquad
=================================================

.. py:class:: mindspore.dataset.audio.transforms.BassBiquad(sample_rate, gain, central_freq=100.0, Q=0.707)

    给音频波形施加低音控制效果，即双极点低频搁架滤波器。

    低频搁架滤波器能够通过所有频率，但将低于搁架的频率提升或衰减指定量，其系统函数为：

    .. math::
        H(s) = A\frac{s^2 + \frac{\sqrt{A}}{Q}s + A}{As^2 + \frac{\sqrt{A}}{Q}s + 1}` 。

    接口实现方式类似于 `SoX库 <http://sox.sourceforge.net/sox.html>`_ 。

    .. note:: 待处理音频维度需为(..., time)。

    **参数：**

    - **sample_rate** (int) - 采样频率（单位：Hz），不能为零。
    - **gain** (float) - 期望提升（或衰减）的音频增益（单位：dB）。
    - **central_freq** (float, 可选) - 中心频率（单位：Hz），默认值：100.0。
    - **Q** (float, 可选) - `品质因子 <https://zh.wikipedia.org/wiki/%E5%93%81%E8%B3%AA%E5%9B%A0%E5%AD%90>`_ ，能够反映带宽与采样频率和中心频率的关系，取值范围为(0, 1]，默认值：0.707。
    
    **样例：**

    >>> import numpy as np
    >>>
    >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
    >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    >>> transforms = [audio.BassBiquad(44100, 100.0)]
    >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
