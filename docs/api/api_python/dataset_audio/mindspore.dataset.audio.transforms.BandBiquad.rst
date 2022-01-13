mindspore.dataset.audio.transforms.BandBiquad
=================================================

.. py:class:: mindspore.dataset.audio.transforms.BandBiquad(sample_rate, central_freq, Q=0.707, noise=False)

    给音频波形施加双极点带通滤波器。

    带通滤波器的频率响应在中心频率附近呈对数下降，下降的斜率由带宽决定，频带两端处输出音频的幅度将是原始幅度的一半。

    接口实现方式类似于 `SoX库 <http://sox.sourceforge.net/sox.html>`_ 。

    .. note:: 待处理音频维度需为(..., time)。

    **参数：**

    - **sample_rate** (int) - 采样频率（单位：Hz），不能为零。
    - **central_freq** (float) - 中心频率（单位：Hz）。
    - **Q** (float, 可选) - `品质因子 <https://zh.wikipedia.org/wiki/%E5%93%81%E8%B3%AA%E5%9B%A0%E5%AD%90>`_ ，能够反映带宽与采样频率和中心频率的关系，取值范围为(0, 1]，默认值：0.707。
    - **noise** (bool, 可选) - 若为True，则使用非音调音频（如打击乐）模式；若为False，则使用音调音频（如语音、歌曲或器乐）模式，默认值：False。

    **样例：**

    >>> import numpy as np
    >>>
    >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
    >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    >>> transforms = [audio.BandBiquad(44100, 200.0)]
    >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
