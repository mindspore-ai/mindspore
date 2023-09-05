mindspore.ops.UniformReal
=========================

.. py:class:: mindspore.ops.UniformReal(seed=0, seed2=0)

    产生随机的浮点数，均匀分布在[0, 1)范围内。

    .. note::
        - 随机种子：通过一些复杂的数学算法，可以得到一组有规律的随机数，而随机种子就是这个随机数的初始值。随机种子相同，得到的随机数就不会改变。
        - 全局的随机种子和算子层的随机种子都没设置或都设置为0：完全随机。
        - 全局的随机种子设置了，算子层的随机种子未设置：采用全局的随机种子和0拼接。
        - 全局的随机种子未设置，算子层的随机种子设置了：使用0和算子层的随机种子拼接。
        - 全局的随机种子和算子层的随机种子都设置了：全局的随机种子和算子层的随机种子拼接。
        - 目前在Ascend平台上不支持 `shape` 为Tensor，CPU/GPU平台支持。当输入为Tensor的时候，支持的数据类型：
          - GPU：int32、int64。
          - CPU：int16、int32、int64。

    参数：
        - **seed** (int，可选) - 算子层的随机种子，用于生成随机数。必须是非负的。默认值： ``0`` 。
        - **seed2** (int，可选) - 全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。必须是非负的。默认值： ``0`` 。

    输入：
        - **shape** (Union[tuple, Tensor]) - 待生成的Tensor的shape。只支持常量值。

    输出：
        Tensor。它的shape为输入 `shape`。数据类型为float32。

    异常：
        - **TypeError** - `seed` 或 `seed2` 不是int。
        - **TypeError** - `shape` 不是tuple或Tensor。
        - **ValueError** - `shape` 不是常量值。
