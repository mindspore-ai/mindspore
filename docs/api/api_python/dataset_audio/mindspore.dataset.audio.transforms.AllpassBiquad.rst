mindspore.dataset.audio.transforms.AllpassBiquad
=================================================

.. py:class:: mindspore.dataset.audio.transforms.AllpassBiquad(sample_rate, central_freq, Q=0.707)

    给音频波形施加双极点全通滤波器，其中心频率和带宽由入参指定。

    全通滤波器能够改变音频频率与相位的关系，而不改变频率与幅度的关系，其系统函数为：

    .. math::
        H(s) = \frac{s^2 - \frac{s}{Q} + 1}{s^2 + \frac{s}{Q} + 1}

    接口实现方式类似于 `SoX库 <http://sox.sourceforge.net/sox.html>`_ 。
    
    .. note:: 待处理音频维度需为(..., time)。

    **参数：**

    - **sample_rate** (int) - 采样频率（单位：Hz），不能为零。
    - **central_freq** (float) - 中心频率（单位：Hz）。
    - **Q** (float, 可选) - `品质因子 <https://zh.wikipedia.org/wiki/%E5%93%81%E8%B3%AA%E5%9B%A0%E5%AD%90>`_ ，能够反映带宽与采样频率和中心频率的关系，取值范围为(0, 1]，默认值：0.707。

    **样例：**

    >>> import numpy as np
    >>>
    >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
    >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    >>> transforms = [audio.AllpassBiquad(44100, 200.0)]
    >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
