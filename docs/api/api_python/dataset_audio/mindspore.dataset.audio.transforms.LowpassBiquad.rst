mindspore.dataset.audio.transforms.LowpassBiquad
=================================================

.. py:class:: mindspore.dataset.audio.transforms.LowpassBiquad(sample_rate, cutoff_freq, Q=0.707)

    给形如(..., time)维度的音频波形施加双极低通滤波器。实现方式类似于SoX库。

    **参数：**

    - **sample_rate** (int) - 采样率，例如44100 (Hz)，不能为零。
    - **cutoff_freq** (float) - 中心频率（单位：Hz）。
    - **Q** (float, optional) - 品质因子，参考 https://en.wikipedia.org/wiki/Q_factor，取值范围(0, 1]（默认值为0.707）。

    **样例：**

    >>> import numpy as np
    >>>
    >>> waveform = np.array([[0.8236, 0.2049, 0.3335], [0.5933, 0.9911, 0.2482],
    ...                      [0.3007, 0.9054, 0.7598], [0.5394, 0.2842, 0.5634], [0.6363, 0.2226, 0.2288]])
    >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    >>> transforms = [audio.LowpassBiquad(4000, 1500, 0.7)]
    >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
