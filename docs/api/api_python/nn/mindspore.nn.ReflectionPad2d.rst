mindspore.nn.ReflectionPad2d
============================

.. py:class:: mindspore.nn.ReflectionPad2d(padding)

    根据 `padding` 对输入 `x` 进行填充。

    **参数：**

    - **padding** (tuple/int) - 填充大小， 如果输入为int， 则对所有边界进行相同大小的填充； 如果是tuple，则顺序为(pad_left, pad_right, pad_up, pad_down)。

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

    Tensor，填充后的Tensor, shape为 :math:`(C, H_out, W_out)` 或 :math:`(N, C, H_out, W_out)`。其中 :math:`H_out = H_in + pad_up + pad_down`,:math:`W_out = W_in + pad_left + pad_right`。

    **异常：**

    - **TypeError** - `padding` 不是tuple或int。
    - **TypeError** - `padding` 中存在不是int的元素。
    - **ValueError** - `padding` 是tuple，且长度不能被2整除。
    - **ValueError** - `padding` 是tuple，且存在负数。
    - **ValueError** - `padding` 是tuple，且长度和tensor的维度不匹配。
