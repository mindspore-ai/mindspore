mindspore.ops.AvgPool
======================

.. py:class:: mindspore.ops.AvgPool(kernel_size=1, strides=1, pad_mode='valid', data_format='NCHW')

    对输入的多维数据进行二维平均池化运算。

    更多参考详见 :func:`mindspore.ops.avg_pool2d`。

    参数：
        - **kernel_size** (Union[int, tuple[int]]) - 指定池化核尺寸大小，可以是单个整数表示池化核高度和宽度，或者整数tuple分别表示池化核高度和宽度。默认值： ``1`` 。
        - **strides** (Union[int, tuple[int]]) - 池化操作的移动步长，可以是单个整数表示高度和宽度方向的移动步长，或者整数tuple分别表示高度和宽度方向的移动步长。默认值： ``1`` 。
        - **pad_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"same"`` 或 ``"valid"``。默认值： ``"valid"`` 。

          - ``"same"``：在输入的四周填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周，若为奇数，多余的填充量将补充在底部/右侧。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大高度和宽度，不能构成一个完整stride的额外的像素将被丢弃。

        - **data_format** (str，可选) - 指定输入和输出的数据格式。取值为 ``'NHWC'`` 或 ``'NCHW'`` 。默认值： ``'NCHW'`` 。

    输入：
        - **x** (Tensor) - 输入shape为 :math:`(N, C_{in}, H_{in}, W_{in})` 的Tensor。支持的数据类型：float16、float32、float64。

    输出：
        Tensor，shape为 :math:`(N, C_{out}, H_{out}, W_{out})` 。

    异常：
        - **TypeError** - `kernel_size` 或 `strides` 既不是int也不是tuple。
        - **TypeError** - `x` 的数据类型不是float16、float32或float64。
        - **ValueError** - `pad_mode` 既不是'valid'，也不是'same'，不区分大小写。
        - **ValueError** - `data_format` 既不是'NCHW'也不是'NHWC'。
        - **ValueError** - `kernel_size` 或 `strides` 小于1。
        - **ValueError** - `x` 的shape长度不等于4。
