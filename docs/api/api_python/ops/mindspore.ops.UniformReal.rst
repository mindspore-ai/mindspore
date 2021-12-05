mindspore.ops.UniformReal
=========================

.. py:class:: mindspore.ops.UniformReal(seed=0, seed2=0)

    产生随机的浮点数i，均匀分布在[0，1)范围内。

    **参数：**

    - **seed** (int) - 算子层的随机种子，用于生成随机数。必须是非负的。默认值：0。
    - **seed2** (int)：全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。必须是非负的。默认值：0。

    **输入：**

    - **shape** (tuple) - 待生成的随机Tensor的shape。只支持常量值。

    **输出：**

    Tensor。它的shape为输入 `shape` 表示的值。数据类型为float32。

    **异常：**

    - **TypeError** - `seed` 和 `seed2` 都不是int。
    - **TypeError** - `shape` 不是tuple。
    - **ValueError** - `shape` 不是常量值。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> shape = (2, 2)
    >>> uniformreal = ops.UniformReal(seed=2)
    >>> output = uniformreal(shape)
    >>> result = output.shape
    >>> print(result)
    (2, 2)
