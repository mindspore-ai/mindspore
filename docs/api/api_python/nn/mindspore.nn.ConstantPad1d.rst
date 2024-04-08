mindspore.nn.ConstantPad1d
==========================

.. py:class:: mindspore.nn.ConstantPad1d(padding, value)

    将给定的常量填充到多维输入数据的最后一维。

    参数：
        - **padding** (Union[int, tuple]) - 指定填充的大小。如果 `padding` 的类型为int，则在输入最后一维的前后均填充 `padding` 大小，如果padding的类型为tuple，形如(padding_0, padding_1)，那么输入 `x` 对应输出的最后一维的shape为 :math:`padding\_0 + x.shape[-1] + padding\_1` ，输出的其余维度与输入保持一致。在Ascend后端运行时，不支持 `padding` 包含负值情况。
        - **value** (Union[int, float]) - 填充值。

    输入：
        - **x** (Tensor) - 输入Tensor，shape为 :math:`(N, *)`，其中 :math:`*` 表示任意维度。在Ascend后端运行时，不支持维度大于5。

    返回：
        Tensor，填充后的Tensor。

    异常：
        - **TypeError** - `padding` 既不是tuple或者int。
        - **TypeError** - `value` 既不是int，也不是float。
        - **ValueError** - tuple类型的 `padding` 长度不等于2。
        - **ValueError** - 填充后输出的维度不是正数。
        - **ValueError** - 在Ascend后端运行时，`x` 的维度大于5。
        - **ValueError** - 在Ascend后端运行时，`padding` 中包含负值。
