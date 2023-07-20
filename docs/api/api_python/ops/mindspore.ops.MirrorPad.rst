mindspore.ops.MirrorPad
=======================

.. py:class:: mindspore.ops.MirrorPad(mode='REFLECT')

    通过指定的填充模式和大小对输入Tensor进行填充。

    参数：
        - **mode** (str，可选) - 指定填充模式的可选字符串。可选值为： ``'REFLECT'`` 和 ``'SYMMETRIC'`` 。默认值： ``'REFLECT'`` 。
          当采样grid超出输入Tensor的边界时，各种填充模式效果如下：

          - ``'REFLECT'`` ：使用零填充输入Tensor。例如，向 [1, 2, 3, 4] 的两边分别填充2个元素，结果为 [3, 2, 1, 2, 3, 4, 3, 2]。
          - ``'SYMMETRIC'`` ：使用Tensor边缘上像素的值填充输入Tensor。例如，向 [1, 2, 3, 4] 的两边分别填充2个元素，结果为 [2, 1, 1, 2, 3, 4, 4, 3]。

    输入：
        - **input_x** (Tensor) - shape: :math:`(N, *)` ，其中 :math:`*` 表示任何数量的附加维度。
        - **paddings** (Tensor) - shape为 :math:`(N, 2)` 的矩阵。N为输入Tensor的秩。int类型。
          对于输入的第 `D` 个维度， `paddings[D, 0]` 表示需在输入第 `D` 维头部填充的数量， `paddings[D, 1]` 表示需在输入第 `D` 维尾部填充的数量。

    输出：
        填充后的Tensor。

        - 如果设置 `mode` 为 ``'REFLECT'`` ，将使用对称轴对称复制的方式来进行填充。
          如果 `input_x` 为[[1,2,3]，[4,5,6]，[7,8,9]]， `paddings` 为[[1,1], [2,2]]，则输出为[[6,5,4,5,6,5,4]，[3,2,1,2,3,2,1]，[6,5,4,5,6,5,4]，[9,8,7,8,9,8,7]，[6,5,4,5,6,5,4]]。
          更直观的理解请参见下面的样例。
        - 如果 `mode` 为 ``'SYMMETRIC'`` ，则填充方法类似于 ``'REFLECT'`` 。它也会根据对称轴复制，但是也包括对称轴。如果 `input_x` 为[[1,2,3],[4,5,6],[7,8,9]]， `paddings` 为[[1,1], [2,2]]，则输出为[[2,1,1,2,3,3,2]，[2,1,1,2,3,3,2]，[5,4,4,5,6,6,5]，[8,7,7,8,9,9,8]，[8,7,7,8,9,9,8]]。
          更直观的理解请参见下面的样例。

    异常：
        - **TypeError** - `input_x` 或 `padings` 不是Tensor。
        - **TypeError** - `mode` 不是str。
        - **ValueError** - `paddings.size` 不等于2 * len(`input_x`)。
