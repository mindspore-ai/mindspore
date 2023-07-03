mindspore.ops.RandomPoisson
============================

.. py:class:: mindspore.ops.RandomPoisson(seed=0, seed2=0, dtype=mstype.int64)

    根据离散概率密度函数分布生成随机非负数浮点数i。函数定义如下：

    .. math::
        \text{P}(i|μ) = \frac{\exp(-μ)μ^{i}}{i!}

    .. note::
        - 随机种子：通过一些复杂的数学算法，可以得到一组有规律的随机数，而随机种子就是这个随机数的初始值。随机种子相同，得到的随机数就不会改变。
        - 全局的随机种子和算子层的随机种子都没设置或都设置为0：完全随机。
        - 全局的随机种子设置了，算子层的随机种子未设置：采用全局的随机种子和0拼接。
        - 全局的随机种子未设置，算子层的随机种子设置了：使用0和算子层的随机种子拼接。
        - 全局的随机种子和算子层的随机种子都设置了：全局的随机种子和算子层的随机种子拼接。

    参数：
        - **seed** (int，可选) - 算子层的随机种子，用于生成随机数。必须是非负的。默认值： ``0`` 。
        - **seed2** (int，可选) - 全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。必须是非负的。默认值： ``0`` 。
        - **dtype** (mindspore.dtype，可选) - 输出数据类型， 默认值： ``mstype.int64`` 。

    输入：
        - **shape** (tuple) - 待生成的随机Tensor的shape，是一个一维Tensor。数据类型为int32或int64。
        - **rate** (Tensor) - `rate` 为Poisson分布的μ参数，决定数字的平均出现次数。数据类型是其中之一：[float16, float32, float64, int32, int64]。

    输出：
        Tensor。shape是 :math:`(*shape, *rate.shape)` ，数据类型由参数 `dtype` 指定。

    异常：
        - **TypeError** - `shape` 不是Tensor或数据类型不是int32或int64。
        - **TypeError** - `dtype` 数据类型不是int32或int64。
        - **ValueError** - `shape` 不是一维Tensor。
        - **ValueError** - `shape` 的元素存在负数。
