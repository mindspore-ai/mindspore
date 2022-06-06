mindspore.nn.ReflectionPad2d
============================

.. py:class:: mindspore.nn.ReflectionPad2d(paddings)

    根据 `paddings` 对输入 `x` 进行填充。

    **参数：**

    - **paddings** (tuple/int) - 填充大小，如果输入为integer, 则对所有边界进行相同大小的padding，如果是tuple，则顺序为(pad_left, pad_right, pad_up, pad_down)。

      .. code-block::

          # 假设参数和输入如下：
          paddings = (1, 1, 2, 0).
          x = [[[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]].
          # `x` 的第一个维度为1， 第二个维度为1， 第三个维度为3，第四个维度为3。
          # 输出的第一个维度不变。
          # 输出的第二个维度不变。
          # 输出的第三个维度为H_out = H_in + pad_up + pad_down = 3 + 1 + 1 = 5。
          # 输出的第四个维度为W_out = W_in + pad_left + pad_right = 3 + 2 + 0 = 5。
          # 所以最终的输出shape为(1, 1, 5, 5)

    **输入：**

    - **x** (Tensor) - 输入Tensor, shape为 :math:`(C, H_in, W_in)` 或 :math:`(N, C, H_in, W_in)`。

    **输出：**

    Tensor，填充后的Tensor, shape为 :math:`(C, H_out, W_out)` 或 :math:`(N, C, H_out, W_out)`。其中 :math:`H_out = H_in + pad_up + pad_down`,:math:`W_out = W_in + pad_left + pad_right, H_out = H_in`

    - 对 `x` 使用对称轴进行对称复制的方式进行填充（复制时不包括对称轴）。例如 `x` 为[[[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]]， `paddings` 为(1, 1, 2, 0)，则输出为[[[[7., 6., 7., 8., 7.], [4., 3., 4., 5., 4.], [1., 0., 1., 2., 1.], [4., 3., 4., 5., 4.], [7., 6., 7., 8., 7.]]]]。

    **异常：**

    - **TypeError** - `padding` 不是tuple或integer。
    - **ValueError** - `padding` 中存在不是integer的元素
    - **ValueError** - `padding` 是tuple，且长度不能被2整除。
    - **ValueError** - `padding` 是tuple，且存在负数。
    - **ValueError** - `padding` 是tuple，且和tensor的维度不匹配。
