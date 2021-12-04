mindspore.dataset.audio.transforms.BassBiquad
=================================================

.. py:class:: mindspore.dataset.audio.transforms.BassBiquad(sample_rate, gain, central_freq=100.0, Q=0.707)

    给形如(..., time)维度的音频波形施加低音控制效果。

    **参数：**

    - **sample_rate** (int) - 采样率，例如44100 (Hz)，不能为零。
    - **gain** (float) - 期望提升（或衰减）的音频增益，单位为dB。
    - **central_freq** (float) - 中心频率（单位：Hz）。
    - **Q** (float, optional) - 品质因子，参考 https://en.wikipedia.org/wiki/Q_factor，取值范围(0, 1]（默认值为0.707）。

    **样例：**

    >>> import numpy as np
    >>>
    >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
    >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    >>> transforms = [audio.BassBiquad(44100, 100.0)]
    >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
