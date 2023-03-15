mindspore.ops.argmax
====================

.. py:function:: mindspore.ops.argmax(input, dim=None, keepdim=False)

    返回输入Tensor在指定轴上的最大值索引。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **dim** (Union[int, None]，可选) - 指定计算轴。如果是None，将会返回扁平化Tensor在指定轴上的最大值索引。默认值：None。
        - **keepdim** (bool，可选) - 输出Tensor是否保留指定轴。如果 `axis` 是None，忽略该选项。默认值： False。

    返回：
        Tensor，输出为指定轴上输入Tensor最大值的索引。

    异常：
        - **ValueError** - 如果入参 `axis` 的设定值超出了范围。
