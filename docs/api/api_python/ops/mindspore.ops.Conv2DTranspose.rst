mindspore.ops.Conv2DTranspose
==============================

.. py:class:: mindspore.ops.Conv2DTranspose(out_channel, kernel_size, pad_mode="valid", pad=0, pad_list=None, mode=1, stride=1, dilation=1, group=1, data_format='NCHW')

    计算二维转置卷积，也称为反卷积，实际不是真正的反卷积。因为它不能完全的恢复输入矩阵的数据，但能恢复输入矩阵的形状。

    参数：
        - **out_channel** (int) - 输出的通道数。
        - **kernel_size** (Union[int, tuple[int]]) - 卷积核的大小。
        - **pad_mode** (str，可选) - 指定填充模式，填充值为0。可选值为 ``"same"`` ， ``"valid"`` 或 ``"pad"`` 。默认值： ``"valid"`` 。

          - ``"same"``：在输入的四周填充，使得当 `stride` 为 ``1`` 时，输入和输出的shape一致。待填充的量由算子内部计算，若为偶数，则均匀地填充在四周，若为奇数，多余的填充量将补充在底部/右侧。如果设置了此模式， `pad` 必须为0。
          - ``"valid"``：不对输入进行填充，返回输出可能的最大高度和宽度，不能构成一个完整stride的额外的像素将被丢弃。如果设置了此模式， `pad` 必须为0。
          - ``"pad"``：对输入填充指定的量。在这种模式下，在输入的高度和宽度方向上填充的量由 `pad` 参数指定。如果设置此模式， `pad` 必须大于或等于0。
            请参考 :class:`mindspore.nn.Conv2dTranspose` 了解更多 `pad_mode` 的使用规则。

        - **pad** (Union[int, tuple[int]]) - 指定要填充的填充值。默认值： ``0`` 。如果 `pad` 是整数，则顶部、底部、左侧和右侧的填充都等于 `pad` 。如果 `pad` 是四个整数的tuple，则顶部、底部、左侧和右侧的填充分别等于pad[0]、pad[1]、pad[2]和pad[3]。
        - **pad_list** (Union[str, None]) - 卷积填充方式，如（顶部、底部、左、右）。默认值： ``None`` ，表示不使用此参数。
        - **mode** (int) - 指定不同的卷积模式。当前未使用该值。默认值： ``1`` 。
        - **stride** (Union[int, tuple[int]]) - 卷积核移动的步长。默认值： ``1`` 。
        - **dilation** (Union[int, tuple[int]]) - 卷积核膨胀尺寸。默认值： ``1`` 。
        - **group** (int) - 将过滤器拆分为组。默认值： ``1`` 。
        - **data_format** (str) - 输入和输出的数据格式。它应该是 ``'NHWC'`` 或 ``'NCHW'`` ，默认值是 ``'NCHW'`` 。

    输入：
        - **dout** (Tensor) - 卷积操作的输出的梯度Tensor。shape： :math:`(N, C_{out}, H_{out}, W_{out})` 。
        - **weight** (Tensor) - 设置卷积核大小为 :math:`(K_1, K_2)` ，然后shape为 :math:`(C_{out}, C_{in}, K_1, K_2)` 。
        - **input_size** (Tensor) - 输入的shape，shape的格式为 :math:`(N, C_{in}, H_{in}, W_{in})` 。

    输出：
        Tensor，卷积操作的输入的梯度Tensor。它的shape与输入相同。

    异常：
        - **TypeError** - 如果 `kernel_size` 、 `stride` 、 `pad` 或 `diation` 既不是int也不是tuple。
        - **TypeError** - 如果 `out_channel` 或 `group` 不是int。
        - **ValueError** - 如果 `kernel_size` 、 `stride` 或 `dlation` 小于1。
        - **ValueError** - 如果 `pad_mode` 不是'same'、'valid'或'pad'。
        - **ValueError** - 如果 `padding` 是长度不等于4的tuple。
        - **ValueError** - 如果 `pad_mode` 不等于'pad'，`pad` 不等于（0,0,0,0）。
        - **ValueError** - 如果 `data_format` 既不是'NCHW'也不是'NHWC'。
