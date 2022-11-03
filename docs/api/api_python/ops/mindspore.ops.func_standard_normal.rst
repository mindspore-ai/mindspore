mindspore.ops.standard_normal
=============================

.. py:function:: mindspore.ops.standard_normal(shape, seed=0, seed2=0)

    根据标准正态（高斯）随机数分布生成随机数。

    返回具有给定shape的Tensor，其中的随机数从平均值为0、标准差为1的标准正态分布中取样。

    .. math::
        f(x)=\frac{1}{\sqrt{2 \pi}} e^{\left(-\frac{x^{2}}{2}\right)}

    参数：
        - **shape** (Union[tuple, Tensor]) - 待生成的Tensor的shape。当为tuple类型时，只支持常量值；当为Tensor类型时，支持动态Shape。
        - **seed** (int) - 随机种子，非负值。默认值：0。
        - **seed2** (int) - 随机种子2，用来防止随机种子冲突，非负值。默认值：0。

    返回：
        Tensor。shape为输入 `shape` 。数据类型支持float32。

    异常：
        - **TypeError** - `seed` 或 `seed2` 不是int类型。
        - **TypeError** - `shape` 既不是tuple，也不是Tensor。
        - **ValueError** - `seed` 或 `seed2` 不是非负的int。
        - **ValueError** - `shape` 为tuple时，包含非正的元素。
