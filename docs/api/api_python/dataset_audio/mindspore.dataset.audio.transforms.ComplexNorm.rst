mindspore.dataset.audio.transforms.ComplexNorm
=================================================

.. py:class:: mindspore.dataset.audio.transforms.ComplexNorm(power=1.0)

    计算复数序列的范数。

    .. note:: 待处理音频维度需为(..., complex=2)，其中第0维代表实部，第1维代表虚部。

    **参数：**

    - **power** (float, 可选) - 范数的幂，取值必须非负，默认值：1.0。

    **样例：**

    >>> import numpy as np
    >>>
    >>> waveform = np.random.random([2, 4, 2])
    >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    >>> transforms = [audio.ComplexNorm()]
    >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
