mindspore.ops.DataFormatVecPermute
===================================

.. py:class:: mindspore.ops.DataFormatVecPermute(src_format='NHWC', dst_format='NCHW')

    将输入Tensor的数据格式从 `src_format` 转换为 `dst_format` ，通过重新排列其维度来实现。

    参数：
        - **src_format** (str, 可选) - 初始数据排列格式，可以为'NHWC'和'NCHW'之一。默认值：'NHWC'。
        - **dst_format** (str, 可选) - 目标数据排列格式，可以为'NHWC'和'NCHW'之一。默认值：'NCHW'。

    输入：
        - **input_x** (Tensor) - shape为 :math:`(4, )` 或 :math:`(4, 2)` 的输入Tensor。数据类型为int32或int64。

    输出：
        与 `input_x` 的shape和数据类型一致的Tensor。

    异常：
        - **TypeError** - 输入 `input_x` 不是Tensor。
        - **TypeError** -  `input_x` 的数据类型不是int32或int64。
        - **ValueError** -  `input_x` 的shape不为 :math:`(4, )` or :math:`(4, 2)` 。
        - **ValueError** -  `src_format` 或 `dst_format` 不是'NHWC'或'NCHW'之一。
