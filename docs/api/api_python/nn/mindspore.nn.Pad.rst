mindspore.nn.Pad
=================

.. py:class:: mindspore.nn.Pad(paddings, mode="CONSTANT")

    根据 `paddings` 和 `mode` 对输入进行填充。

    参数：
        - **paddings** (tuple) - 填充大小，其shape为 :math:`(N, 2)` ，N是输入数据的维度，填充的元素为int类型。对于 `x` 的第 `D` 个维度，paddings[D, 0]表示要在输入Tensor的第 `D` 个维度之前扩展的大小，paddings[D, 1]表示在输入Tensor的第 `D` 个维度后面要扩展的大小。每个维度填充后的大小为： :math:`paddings[D, 0] + input\_x.dim\_size(D) + paddings[D, 1]`。

          .. code-block::

              # 假设参数和输入如下：
              mode = "CONSTANT".
              paddings = [[1,1], [2,2]].
              x = [[1,2,3], [4,5,6], [7,8,9]].
              # `x` 的第一个维度为3， `x` 的第二个维度为3。
              # 根据以上公式可得：
              # 输出的第一个维度是paddings[0][0] + 3 + paddings[0][1] = 1 + 3 + 1 = 5。
              # 输出的第二个维度是paddings[1][0] + 3 + paddings[1][1] = 2 + 3 + 2 = 7。
              # 所以最终的输出shape为(5, 7)

        - **mode** (str) - 指定填充模式。取值为 ``"CONSTANT"`` (常数填充) ，``"REFLECT"`` (反射填充) ，``"SYMMETRIC"`` (对称填充) 。默认值：``"CONSTANT"`` 。

    输入：
        - **x** (Tensor) - 输入Tensor。

    输出：
        Tensor，填充后的Tensor。

        - 如果 `mode` 为"CONSTANT"， `x` 使用0进行填充。例如， `x` 为[[1,2,3]，[4,5,6]，[7,8,9]]， `paddings` 为[[1,1]，[2,2]]，则输出为[[0,0,0,0,0,0,0]，[0,0,1,2,3,0,0]，[0,0,4,5,6,0,0]，[0,0,7,8,9,0,0]，[0,0,0,0,0,0,0]]。
        - 如果 `mode` 为"REFLECT"， `x` 使用对称轴进行对称复制的方式进行填充（复制时不包括对称轴）。例如 `x` 为[[1,2,3]，[4,5,6]，[7,8,9]]， `paddings` 为[[1,1]，[2,2]]，则输出为[[6,5,4,5,6,5,4]，[3,2,1,2,3,2,1]，[6,5,4,5,6,5,4]，[9,8,7,8,9,8,7]，[6,5,4,5,6,5,4]]。
        - 如果 `mode` 为"SYMMETRIC"，此填充方法类似于"REFLECT"。也是根据对称轴填充，包含对称轴。例如 `x` 为[[1,2,3]，[4,5,6]，[7,8,9]]， `paddings` 为[[1,1]，[2,2]]，则输出为[[2,1,1,2,3,3,2]，[2,1,1,2,3,3,2]，[5,4,4,5,6,6,5]，[8,7,7,8,9,9,8]，[8,7,7,8,9,9,8]]。

    异常：
        - **TypeError** - `paddings` 不是tuple。
        - **ValueError** - `paddings` 的长度超过4或其shape不是 :math:`(N, 2)` 。
        - **ValueError** - `mode` 不是 ``'CONSTANT'`` ， ``'REFLECT'`` 或 ``'SYMMETRIC'`` 。