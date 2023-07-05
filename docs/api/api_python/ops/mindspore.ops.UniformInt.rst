mindspore.ops.UniformInt
========================

.. py:class:: mindspore.ops.UniformInt(seed=0, seed2=0)

    根据均匀分布在区间 `[minval, maxval)` 中生成随机数。离散概率函数定义如下：

    .. math::
        \text{P}(i|a,b) = \frac{1}{b-a+1},

    其中 :math:`a` 为分布区间的最小值 `minval` ， :math:`b` 为分布区间的最大值 `maxval` 。

    .. note::
        - `minval` 中的数值在广播后必须严格小于 `maxval` 。
        - 随机种子：通过一些复杂的数学算法，可以得到一组有规律的随机数，而随机种子就是这个随机数的初始值。随机种子相同，得到的随机数就不会改变。
        - 全局的随机种子和算子层的随机种子都没设置或都设置为0：完全随机。
        - 全局的随机种子设置了，算子层的随机种子未设置：采用全局的随机种子和0拼接。
        - 全局的随机种子未设置，算子层的随机种子设置了：使用0和算子层的随机种子拼接。
        - 全局的随机种子和算子层的随机种子都设置了：全局的随机种子和算子层的随机种子拼接。

    参数：
        - **seed** (int，可选) - 算子层的随机种子，用于生成随机数。必须是非负的。默认值： ``0`` 。
        - **seed2** (int，可选) - 全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。必须是非负的。默认值： ``0`` 。

    输入：
        - **shape** (Union[tuple, Tensor]) - 目标Tensor的shape。只允许常量值。
        - **minval** (Tensor) - 分布参数， :math:`a` 。
          决定可能生成的最小值，数据类型为int32。需为标量。
        - **maxval** (Tensor) - 分布参数， :math:`b` 。
          决定生成随机数的上限，数据类型为int32。需为标量。

    输出：    
        Tensor。shape为输入 `shape` ，数据类型支持int32。

    异常：
        - **TypeError** - `seed` 或 `seed2` 不是int类型。
        - **TypeError** - `shape` 不是tuple或Tensor。
        - **TypeError** - `minval` 或 `maxval` 不是Tensor。
        - **ValueError** - `shape` 不是常量值。
