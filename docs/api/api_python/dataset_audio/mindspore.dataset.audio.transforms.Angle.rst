mindspore.dataset.audio.transforms.Angle
=================================================

.. py:class:: mindspore.dataset.audio.transforms.Angle

    计算复数序列的角度。

    .. note:: 待处理音频维度需为(..., complex=2)，其中第0维代表实部，第1维代表虚部。

    **样例：**

    >>> import numpy as np
    >>>
    >>> waveform = np.array([[1.43, 5.434], [23.54, 89.38]])
    >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    >>> transforms = [audio.Angle()]
    >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
