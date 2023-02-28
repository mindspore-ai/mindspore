mindspore.ops.UniformInt
========================

.. py:class:: mindspore.ops.UniformInt(seed=0, seed2=0)

    根据均匀分布在区间 `[minval, maxval)` 中生成随机数。离散概率函数定义如下：

    .. math::
        \text{P}(i|a,b) = \frac{1}{b-a+1},

    其中 :math:`a` 为分布区间的最小值 `minval` ， :math:`b` 为分布区间的最大值 `maxval` 。

    .. note::
        - `minval` 中的数值在广播后必须严格小于 `maxval` 。
        - 如果 `seed` 和 `seed2` 都没有被赋非零值，则生成一个随机值当做随机种子

    参数：
        - **seed** (int) - 随机种子，非负值。默认值：0。
        - **seed2** (int) - 随机种子2，用来防止随机种子冲突，非负值。默认值：0。

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
