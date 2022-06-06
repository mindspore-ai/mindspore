mindspore.nn.ReflectionPad1d
============================

.. py:class:: mindspore.nn.ReflectionPad1d(paddings)

    根据 `paddings` 对输入 `x` 进行填充。

    **参数：**

    - **paddings** (tuple/int) - 填充大小，如果输入为int, 则对所有边界进行相同大小的padding，如果是tuple，则为(pad_left, pad_right)。

      .. code-block::

          # 假设参数和输入如下：
          paddings = (3, 1)。
          x = [[[0, 1, 2, 3], [4, 5, 6, 7]]].
          # `x` 的第一个维度为1， 第二个维度为2， 第三个维度为4。
          # 输出的第一个维度不变。
          # 输出的第二个维度不变。
          # 输出的第三个维度为W_out = W_in + pad_left + pad_right = 4 + 3 + 1 = 8。
          # 所以最终的输出shape为(1, 2, 8)。

    **输入：**

    - **x** (Tensor) - 输入Tensor, shape为 :math:`(C, W_in)` 或 :math:`(N, C, W_in)`。

    **输出：**

    Tensor，填充后的Tensor, shape为 :math:`(C, W_out)` 或 :math:`(N, C, W_out)`。其中 :math:`W_out = W_in + pad_left + pad_right`

    - 对 `x` 使用对称轴进行对称复制的方式进行填充（复制时不包括对称轴）。例如 `x` 为[[[0, 1, 2, 3], [4, 5, 6, 7]]]， `paddings` 为(3, 1)，则输出为[[[3, 2, 1, 0, 1, 2, 3, 2], [7, 6, 5, 4, 5, 6, 7, 6]]]。

    **异常：**

    - **TypeError** - `padding` 不是tuple或integer。
    - **ValueError** - `padding` 中存在不是integer的元素
    - **ValueError** - `padding` 是tuple，且长度不能被2整除。
    - **ValueError** - `padding` 是tuple，且存在负数。
    - **ValueError** - `padding` 是tuple，且和tensor的维度不匹配。
