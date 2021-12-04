mindspore.dataset.audio.transforms.Contrast
=================================================

.. py:class:: mindspore.dataset.audio.transforms.Contrast(enhancement_amount=75.0)

    给形如(..., time)维度的音频波形施加对比度增强效果。实现方式类似于SoX库。与音频压缩相比，该效果通过修改音频信号使其听起来更响亮。

    **参数：**

    - **enhancement_amount** (float) - 控制音频增益的量。取值范围为[0,100]（默认为75.0）。注意当 `enhancement_amount` 等于0时，对比度增强效果仍然会很显著。

    **样例：**

    >>> import numpy as np
    >>>
    >>> waveform = np.array([[2.716064453125e-03, 6.34765625e-03], [9.246826171875e-03, 1.0894775390625e-02]])
    >>> numpy_slices_dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    >>> transforms = [audio.Contrast()]
    >>> numpy_slices_dataset = numpy_slices_dataset.map(operations=transforms, input_columns=["audio"])
