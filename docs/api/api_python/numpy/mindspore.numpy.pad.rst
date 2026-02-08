mindspore.numpy.pad
=================================

.. py:function:: mindspore.numpy.pad(arr, pad_width, mode='constant', stat_length=None, constant_values=0, end_values=0, reflect_type='even', **kwargs)

    对矩阵进行填充。

    .. note::
        目前，不支持median模式。reflect和symmetric模式只支持GPU端。

    参数：
        - **arr** (Union[list, tuple, Tensor]) - 要填充的矩阵。
        - **pad_width** (Union[int, tuple, list]) - 每个轴的边缘需要填充的值的数目。 ``((before_1, after_1), ... (before_N, after_N))`` 为每个轴创建唯一的填充宽度。 ``((before, after),)`` 为每个轴产生前后相同的 ``pad`` 。 ``(pad,)`` 或int是将所有轴设置为 ``before = after = pad width`` 的快捷设置。
        - **mode** (str, 可选) - 填充的方式。

          选择以下之一：

          - **constant** (默认值): 连续填充相同的值。
          - **edge**: 用 ``arr`` 的边缘值填充。
          - **linear_ramp**: 用边缘递减的方式填充。
          - **maximum**: 最大值填充。
          - **mean**: 均值填充。
          - **median**: 中位数填充。
          - **minimum**: 最小值填充。
          - **reflect**: 以边缘值为轴对称填充。
          - **symmetric**: 沿着边缘值对称填充。
          - **wrap**: 用原数组后面的值填充前面，前面的值填充后面。
          - **empty**: 用未定义值填充。
          - **<function>**: 用自定义函数填充。该自定义函数应修改并返回一个新的一维Tensor。 定义如下： ``padding_func(tensor, iaxis_pad_width, iaxis, kwargs)`` 。

        - **stat_length** (Union[tuple, list, int], 可选) - 选择 ``'maximum', 'mean', 'median', 'minimum'`` 模式时，用于计算统计值的每个轴边缘的值的数量。可以为每个轴指定不同的统计长度，如 ``((before_1, after_1), ... (before_N, after_N))`` ，也可以为所有轴设置唯一的统计长度。 ``((before, after),)`` 为所有轴指定相等的前后统计长度， ``(stat_length,)`` 或int是将所有轴设置为 ``before = after = statistic length`` 的快捷设置。默认值： ``None`` 。
        - **constant_values** (Union[tuple, list, int], 可选) - 选择 ``'constant'`` 模式时，为每个轴设置填充值的常数值。 ``((before_1, after_1), ... (before_N, after_N))`` 可以为每个轴定义唯一的填充常量。 ``((before, after),)`` 为所有轴设置前后相同的填充常量。 ``(constant,)`` 或 ``constant`` 是将所有轴设置为 ``before = after = constant`` 的快捷设置。默认值： ``0`` 。
        - **end_values** (Union[tuple, list, int], 可选) - 选择 ``'linear_ramp'`` 模式时，定义linear_ramp的结束值，这些值将构成填充 ``arr`` 的边缘。 ``((before_1, after_1), ... (before_N, after_N))`` 可为每个轴设置唯一的结束值， ``((before, after),)`` 为所有轴设置前后相同的结束值。``(constant,)`` 或 ``constant`` 是将所有轴设置为 ``before = after = constant`` 的快捷设置。默认值： ``0`` 。
        - **reflect_type** (str, 可选) - 选择'reflect', 'symmetric'模式时，默认为'even'样式，即围绕边缘值未改变的反射。对于'odd'样式，数组的扩展部分是通过从两倍的边缘值中减去反射值来创建的。
        - **kwargs** (anytype, 可选) - 选择 ``<function>`` 模式时，自定义函数使用的关键字参数。

    返回：
        填充后的Tensor，其秩与 ``arr`` 相同，shape根据 ``pad_width`` 增加。

    异常：
        - **TypeError** - 如果输入的参数类型没有按上述内容指定。
        - **ValueError** - 如果 ``mode`` 的输入无法被识别，或如果 ``pad_width`` ,  ``stat_length`` ,  ``constant_values`` ,  ``end_values`` 无法被广播到 ``(arr.ndim, 2)`` 或关键字参数得到了意外的输入。
        - **NotImplementedError** - 如果 ``mode`` 选择为'median'或function。