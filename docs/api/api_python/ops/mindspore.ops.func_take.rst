mindspore.ops.take
=====================

.. py:function:: mindspore.ops.take(input, indices, axis=None, mode='clip')

    在指定维度上获取Tensor中的元素。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **indices** (Tensor) - 待提取的值的shape为 :math:`(N_j...)` 的索引。
        - **axis** (int, 可选) - 在指定维度上选择值。默认值： ``None``。默认场景下，将输入数组展开为一维数组后进行计算。
        - **mode** (str, 可选) - 支持 ``'raise'`` 、 ``'wrap'`` 、 ``'clip'`` 。

          - ``raise``：当``indices``任何一个值超出``input``的索引范围，则会抛出错误。
          - ``wrap``：绕接模式，当``indices``任何一个值超出``input``的索引范围，则会从``input``的开头重新开始取元素。
          - ``clip``：裁剪到范围。 ``clip`` 模式意味着所有过大的索引都会被在指定轴方向上指向最后一个元素的索引替换。注：不支持负数的索引。

          默认值： ``'clip'`` 。

    返回：
        Tensor，索引的结果。

    异常：
        - **ValueError** - `axis` 超出`input`维度范围，或 `mode` 被设置为 ``'raise'`` 、 ``'wrap'`` 和 ``'clip'`` 以外的值。
