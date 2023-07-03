mindspore.ops.TruncatedNormal
==============================

.. py:class:: mindspore.ops.TruncatedNormal(seed=0, seed2=0, dtype=mstype.float32)

    返回一个具有指定shape的Tensor，其数值取自正态分布。

    生成的值符合正态分布。

    .. note::
        - `shape` 所含元素的值必须大于零。输出长度必须不超过1000000。
        - 随机种子：通过一些复杂的数学算法，可以得到一组有规律的随机数，而随机种子就是这个随机数的初始值。随机种子相同，得到的随机数就不会改变。
        - 全局的随机种子和算子层的随机种子都没设置或都设置为0：完全随机。
        - 全局的随机种子设置了，算子层的随机种子未设置：采用全局的随机种子和0拼接。
        - 全局的随机种子未设置，算子层的随机种子设置了：使用0和算子层的随机种子拼接。
        - 全局的随机种子和算子层的随机种子都设置了：全局的随机种子和算子层的随机种子拼接。

    参数：
        - **seed** (int，可选) - 算子层的随机种子，用于生成随机数。必须是非负的。默认值： ``0`` 。
        - **seed2** (int，可选) - 全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。必须是非负的。默认值： ``0`` 。
        - **dtype** (mindspore.dtype，可选) - 指定输出类型。可选值为： ``mindspore.float16`` 、 ``mindspore.float32`` 和 ``mindspore.float64`` 。默认值： ``mindspore.float32`` 。

    输入：
        - **shape** (Tensor) - 生成Tensor的shape。数据类型必须是 ``mindspore.int32`` 或者 ``mindspore.int64`` 。

    输出：
        Tensor，其shape由 `shape` 决定，数据类型由 `dtype` 决定。其值在[-2,2]范围内。

    异常：
        - **TypeError** - `shape` 不是Tensor。
        - **TypeError** - `dtype` 或 `shape` 的数据类型不支持。
        - **TypeError** -  `seed` 不是整数。
        - **ValueError** - `shape` 的元素不全大于零。
        - **ValueError** - `shape` 不是一维Tensor。
        - **ValueError** - 输出Tensor的元素个数大于1000000。

