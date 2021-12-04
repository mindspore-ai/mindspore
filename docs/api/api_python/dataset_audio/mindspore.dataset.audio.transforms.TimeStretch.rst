mindspore.dataset.audio.transforms.TimeStretch
=================================================

.. py:class:: mindspore.dataset.audio.transforms.TimeStretch(hop_length=None, n_freq=201, fixed_rate=None)

    以给定的比例拉伸音频短时傅里叶（STFT）频谱的时域，但不改变音频的音高。

    **参数：**

    - **hop_length** (int, optional) - STFT窗之间每跳的长度，即连续帧之间的样本数（默认为None，取 `n_freq - 1`）。
    - **n_freq** (int, optional) - STFT中的滤波器组数（默认为201）。
    - **fixed_rate** (float, optional) - 频谱在时域加快或减缓的比例（默认为None，取1.0）。

    **样例：**

    >>> import numpy as np
    >>>
    >>> waveform = np.random.random([1, 30])
    >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    >>> transforms = [audio.TimeStretch()]
    >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
