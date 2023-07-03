mindspore.ops.ParameterizedTruncatedNormal
===========================================

.. py:class:: mindspore.ops.ParameterizedTruncatedNormal(seed=0, seed2=0)

    返回一个具有指定shape的Tensor，其数值取自截断正态分布。
    当其shape为 :math:`(batch\_size, *)` 的时候， `mean` 、 `stdevs` 、 `min` 和 `max` 的shape应该为 :math:`()` 或者 :math:`(batch\_size, )` 。

    .. note::
        - 随机种子：通过一些复杂的数学算法，可以得到一组有规律的随机数，而随机种子就是这个随机数的初始值。随机种子相同，得到的随机数就不会改变。
        - 全局的随机种子和算子层的随机种子都没设置或都设置为0：完全随机。
        - 全局的随机种子设置了，算子层的随机种子未设置：采用全局的随机种子和0拼接。
        - 全局的随机种子未设置，算子层的随机种子设置了：使用0和算子层的随机种子拼接。
        - 全局的随机种子和算子层的随机种子都设置了：全局的随机种子和算子层的随机种子拼接。

    参数：
        - **seed** (int，可选) - 算子层的随机种子，用于生成随机数。必须是非负的。默认值： ``0`` 。
        - **seed2** (int，可选) - 全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。必须是非负的。默认值： ``0`` 。

    输入：
        - **shape** (Tensor) - 生成Tensor的shape。shape为 :math:`(batch\_size, *)` ，其中 :math:`*` 为长度不小于1的额外维度。数据类型必须是int32或者int64。
        - **mean** (Tensor) - 截断正态分布均值。 shape为 :math:`()` 或者 :math:`(batch\_size, )` 。数据类型必须是float16、float32或者float64。
        - **stdevs** (Tensor) - 截断正态分布的标准差。其值必须大于零，shape和数据类型与 `mean` 一致。
        - **min** (Tensor) - 最小截断值，shape和数据类型与 `mean` 一致。
        - **max** (Tensor) - 最大截断值，shape和数据类型与 `mean` 一致。

    输出：
        Tensor，其shape由 `shape` 决定，数据类型与 `mean` 一致。

    异常：
        - **TypeError** - `shape` 、 `mean` 、 `stdevs` 、 `min` 和 `max` 数据类型不支持。
        - **TypeError** - `mean` 、 `stdevs` 、 `min` 和 `max` 的shape不一致。
        - **TypeError** - `shape` 、 `mean` 、 `stdevs` 、 `min` 和 `max` 不全是Tensor。
        - **ValueError** -  当其 `shape` 为 :math:`(batch\_size, *)` 时， `mean` 、 `stdevs` 、 `min` 或者 `max` 的shape不是 :math:`()` 或者 :math:`(batch\_size, )` 。
        - **ValueError** - `shape` 的元素不全大于零。
        - **ValueError** - `stdevs` 的值不全大于零。
        - **ValueError** - `shape` 的的元素个数小于2。
        - **ValueError** - `shape` 不是一维Tensor。
