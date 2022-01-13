mindspore.dataset.audio.transforms.TimeStretch
=================================================

.. py:class:: mindspore.dataset.audio.transforms.TimeStretch(hop_length=None, n_freq=201, fixed_rate=None)

    以给定的比例拉伸音频短时傅里叶（Short Time Fourier Transform, STFT）频谱的时域，但不改变音频的音高。

    .. note:: 待处理音频维度需为(..., freq, time, complex=2)，其中第0维代表实部，第1维代表虚部。

    **参数：**

    - **hop_length** (int, 可选) - STFT窗之间每跳的长度，即连续帧之间的样本数，默认值：None，表示取 `n_freq - 1`。
    - **n_freq** (int, 可选) - STFT中的滤波器组数，默认值：201。
    - **fixed_rate** (float, 可选) - 频谱在时域加快或减缓的比例，默认值：None，表示保持原始速率。

    **样例：**

    >>> import numpy as np
    >>>
    >>> waveform = np.random.random([1, 30])
    >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    >>> transforms = [audio.TimeStretch()]
    >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])

    .. image:: api_img/dataset/time_stretch_rate1.5.png

    .. image:: api_img/dataset/time_stretch_original.png

    .. image:: api_img/dataset/time_stretch_rate0.8.png