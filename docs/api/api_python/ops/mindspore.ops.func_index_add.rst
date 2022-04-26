mindspore.ops.index_add
=================

.. py:function:: mindspore.ops.index_add(x, indices, y, axis, use_lock=True, check_index_bound=True)

    将Tensor y加到Parameter x的指定axis轴的指定indices位置。要求axis轴的取值范围为[0,  len(x.dim) - 1]，indices元素的取值范围
    为[0, x.shape[axis] - 1]。

    **参数：**

    - **x** (Parameter) - 被加Parameter。
    - **indices** (Tensor) - 指定Tensor y加到`x`的指定axis轴的指定indices位置。
    - **y** (Tensor) - 与`x`相加的Tensor。
    - **axis** (int) - 指定Tensor y加到`x`的指定axis轴。
    - **use_lock** (bool) - 计算时使用锁。默认值：True。
    - **check_index_bound** (bool) - indices边界检查。默认值：True。

    **返回：**

    相加后的Tensor。shape和数据类型与输入 `x`相同。

    **异常：**

    - **TypeError** - `indices`或者`y`的类型不是Tensor。
    - **ValueError** - `axis`的值超出`x` shape的维度范围。
    - **ValueError** - `x` shape的维度和`y` shape的维度不一致。
    - **ValueError** - `indices` shape的维度不是一维或者`indices` shape的大小与`y` shape在`axis`轴上的大小不一致。
    - **ValueError** - 除`axis`轴外，`x` shape和`y` shape的大小不一致。
