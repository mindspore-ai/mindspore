mindspore.ops.RandomGamma
==========================

.. py:class:: mindspore.ops.RandomGamma(seed=0, seed2=0)

    根据概率密度函数分布生成随机正值浮点数x。

    .. note::
        - 随机种子：通过一些复杂的数学算法，可以得到一组有规律的随机数，而随机种子就是这个随机数的初始值。随机种子相同，得到的随机数就不会改变。
        - 全局的随机种子和算子层的随机种子都没设置：使用默认值当做随机种子。
        - 全局的随机种子设置了，算子层的随机种子未设置：随机生成一个种子和全局的随机种子拼接。
        - 全局的随机种子未设置，算子层的随机种子设置了：使用默认的全局的随机种子，和算子层的随机种子拼接。
        - 全局的随机种子和算子层的随机种子都设置了：全局的随机种子和算子层的随机种子拼接。

    参数：
        - **seed** (int，可选) - 算子层的随机种子，用于生成随机数。必须是非负的。默认值：0。
        - **seed2** (int，可选) - 全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。必须是非负的。默认值：0。

    输入：
        - **shape** (tuple) - 待生成的随机Tensor的shape。只支持常量值。
        - **alpha** (Tensor) - α为RandomGamma分布的shape parameter，主要决定了曲线的形状。其值必须大于0。数据类型为float32。

    输出：
        Tensor。shape是输入 `shape`， `alpha` 相连后的shape。数据类型与 `alpha` 一致。

    异常：
        - **TypeError** - `seed` 或 `seed2` 的数据类型不是int。
        - **TypeError** - `shape` 或 `alpha` 不是Tensor。
        - **TypeError** - `shape` 或 `alpha` 的数据类型不是float32。
        - **ValueError** - `shape` 不是常量值。
