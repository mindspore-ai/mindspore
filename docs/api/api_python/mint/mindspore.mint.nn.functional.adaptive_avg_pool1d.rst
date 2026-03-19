mindspore.mint.nn.functional.adaptive_avg_pool1d
=================================================

.. py:function:: mindspore.mint.nn.functional.adaptive_avg_pool1d(input, output_size)

    对一个多平面输入信号执行一维自适应平均池化。
    也就是说，对于输入任何尺寸，指定输出的尺寸都为L。但是输入和输出特征的数目不会变化。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - adaptive_avg_pool1d的输入，为二维或三维的Tensor。
        - **output_size** (int) - 输出特征图的size。 `output_size` 是一个整数。

    返回：
        Tensor，数据类型与 `input` 相同。

        输出的shape为 `input_shape[:len(input_shape) - 1] + [output_size]` 。

    异常：
        - **ValueError** - 如果 `output_size` 不是整数。
        - **TypeError** - 如果 `input` 不是Tensor。
