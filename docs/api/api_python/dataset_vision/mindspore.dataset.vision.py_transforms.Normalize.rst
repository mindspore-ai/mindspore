mindspore.dataset.vision.py_transforms.Normalize
================================================

.. py:class:: mindspore.dataset.vision.py_transforms.Normalize(mean, std)

    使用指定的均值和标准差，标准化形如(C, H, W)的输入numpy.ndarray图像。

    .. math::

        output_{c} = \frac{input_{c} - mean_{c}}{std_{c}}

    .. note:: 输入图像的像素值需要在[0.0, 1.0]范围内。否则，请先调用 :class:`mindspore.dataset.vision.py_transforms.ToTensor` 进行转换。

    **参数：**

    - **mean** (Union[float, sequence]) - 各通道的像素均值，取值范围为[0.0, 1.0]。若输入浮点型，将为每个通道应用相同的均值；若输入序列，长度应与通道数相等，且对应通道顺序进行排列。
    - **std** (Union[float, sequence]) - 各通道的标准差，取值范围为(0.0, 1.0]。若输入浮点型，将为每个通道应用相同的标准差；若输入序列，长度应与通道数相等，且对应通道顺序进行排列。

    **异常：**

    - **TypeError** - 当输入图像的类型不为 :class:`numpy.ndarray` 。
    - **TypeError** - 当输入图像的维度不为3。
    - **NotImplementedError** - 当输入图像的像素值类型为整型。
    - **ValueError** - 当均值与标准差的长度不相等。
    - **ValueError** - 当均值或标准差的长度即不等于1，也不等于图像的通道数。
