mindspore.common.initializer
=============================

初始化神经元参数。

.. py:class:: mindspore.common.initializer.Initializer(**kwargs)

    初始化器的抽象基类。

    **参数：**

    - **kwargs** (dict) – `Initializer` 的关键字参数。

.. py:class:: mindspore.common.initializer.TruncatedNormal(sigma=0.01)

    生成一个数组用于初始化Tensor，数组中的数值从截断正态分布中采样得到。

    **参数：**

    **sigma** (float) - 截断正态分布的标准差，默认值为0.01。

    **样例：**

    >>> import mindspore
    >>> from mindspore.common.initializer import initializer, TruncatedNormal
    >>> tensor1 = initializer(TruncatedNormal(), [1,2,3], mindspore.float32)
    >>> tensor2 = initializer('truncatedNormal', [1,2,3], mindspore.float32)

.. py:class:: mindspore.common.initializer.Normal(sigma=0.01, mean=0.0)

    生成一个数组用于初始化Tensor，数组中的数值从正态分布N(sigma, mean)中采样得到。

    .. math::
        f(x) =  \frac{1} {\sqrt{2*π} * sigma}exp(-\frac{(x - mean)^2} {2*{sigma}^2})

    **参数：**

    - **sigma** (float) - 正态分布的标准差，默认值为0.01。
    - **mean** (float) - 正态分布的均值，默认值为0.0。

    **样例：**

    >>> import mindspore
    >>> from mindspore.common.initializer import initializer, Normal
    >>> tensor1 = initializer(Normal(), [1,2,3], mindspore.float32)
    >>> tensor2 = initializer('normal', [1,2,3], mindspore.float32)

.. py:class:: mindspore.common.initializer.Uniform(scale=0.07)

    生成一个数组用于初始化Tensor，数组中的数值从均匀分布U(-scale, scale)中采样得到。

    **参数：**

    **scale** (float) - 均匀分布的边界，默认值为0.07。

    **样例：**

    >>> import mindspore
    >>> from mindspore.common.initializer import initializer, Uniform
    >>> tensor1 = initializer(Uniform(), [1,2,3], mindspore.float32)
    >>> tensor2 = initializer('uniform', [1,2,3], mindspore.float32)

.. py:class:: mindspore.common.initializer.HeUniform(negative_slope=0, mode="fan_in", nonlinearity="leaky_relu")

    生成一个数组用于初始化Tensor，数组中的数值从HeKaiming均匀分布U[-boundary,boundary]中采样得到，其中

    .. math::
        boundary = \sqrt{\frac{6}{(1 + a^2) \times \text{fan_in}}}

    是HeUniform分布的边界。

    **参数：**

    - **negative_slope** (int, float, bool) - 本层激活函数的负数区间斜率（仅适用于非线性激活函数"leaky_relu"），默认值为0。
    - **mode** (str) - 可选"fan_in"或"fan_out"，"fan_in"会保留前向传递中权重方差的量级，"fan_out"会保留反向传递的量级，默认为"fan_in"。
    - **nonlinearity** (str) - 非线性激活函数，推荐使用"relu"或"leaky_relu"，默认为"leaky_relu"。

    **样例：**

    >>> import mindspore
    >>> from mindspore.common.initializer import initializer, HeUniform
    >>> tensor1 = initializer(HeUniform(), [1,2,3], mindspore.float32)
    >>> tensor2 = initializer('he_uniform', [1,2,3], mindspore.float32)

.. py:class:: mindspore.common.initializer.HeNormal(negative_slope=0, mode="fan_in", nonlinearity="leaky_relu")

    生成一个数组用于初始化Tensor，数组中的数值从HeKaiming正态分布N(0, sigma^2)中采样得到，其中

    .. math::
        sigma = \frac{gain} {\sqrt{N}}

    其中，gain是一个可选的缩放因子。如果mode是"fan_in"， N是权重Tensor中输入单元的数量，如果mode是"fan_out"， N是权重Tensor中输出单元的数量。

    HeUniform 算法的详细信息，请查看 https://arxiv.org/abs/1502.01852。

    **参数：**

    - **negative_slope** (int, float, bool) - 本层激活函数的负数区间斜率（仅适用于非线性激活函数"leaky_relu"），默认值为0。
    - **mode** (str) - 可选"fan_in"或"fan_out"，"fan_in"会保留前向传递中权重方差的量级，"fan_out"会保留反向传递的量级，默认为"fan_in"。
    - **nonlinearity** (str) - 非线性激活函数，推荐使用"relu"或"leaky_relu"，默认为"leaky_relu"。

    **样例：**

    >>> import mindspore
    >>> from mindspore.common.initializer import initializer, HeNormal
    >>> tensor1 = initializer(HeNormal(), [1,2,3], mindspore.float32)
    >>> tensor2 = initializer('he_normal', [1,2,3], mindspore.float32)

.. py:class:: mindspore.common.initializer.XavierUniform(gain=1)

    生成一个数组用于初始化Tensor，数组中的数值从Xarvier均匀分布U[-boundary,boundary]中采样得到，其中

    .. math::
        boundary = gain * \sqrt{\frac{6}{n_{in} + n_{out}}}

    - :math:`gain` 是一个可选的缩放因子。
    - :math:`n_{in}` 为权重Tensor中输入单元的数量。
    - :math:`n_{out}` 为权重Tensor中输出单元的数量。

    有关 XavierUniform 算法的详细信息，请查看 http://proceedings.mlr.press/v9/glorot10a.html。

    **参数：**

    **gain** (float) - 可选的缩放因子，默认值为1。

    **样例：**

    >>> import mindspore
    >>> from mindspore.common.initializer import initializer, XavierUniform
    >>> tensor1 = initializer(XavierUniform(), [1,2,3], mindspore.float32)
    >>> tensor2 = initializer('xavier_uniform', [1,2,3], mindspore.float32)

.. py:class:: mindspore.common.initializer.One

    生成一个值全为1的常量数组用于初始化Tensor。

    **样例：**

    >>> import mindspore
    >>> from mindspore.common.initializer import initializer, One
    >>> tensor1 = initializer(One(), [1,2,3], mindspore.float32)
    >>> tensor2 = initializer('ones', [1,2,3], mindspore.float32)

.. py:class:: mindspore.common.initializer.Zero

    生成一个值全为0的常量数组用于初始化Tensor。

    **样例：**

    >>> import mindspore
    >>> from mindspore.common.initializer import initializer, Zero
    >>> tensor1 = initializer(Zero(), [1,2,3], mindspore.float32)
    >>> tensor2 = initializer('zeros', [1,2,3], mindspore.float32)

.. py:class:: mindspore.common.initializer.Constant(value)

    生成一个常量数组用于初始化Tensor。

    **参数：**

    **value** (Union[int, numpy.ndarray]) - 用于初始化的常数值或者数组。

    **样例：**

    >>> import mindspore
    >>> from mindspore.common.initializer import initializer
    >>> tensor1 = initializer(0, [1,2,3], mindspore.float32)
    >>> tensor2 = initializer(5, [1,2,3], mindspore.float32)




.. py:function:: mindspore.common.initializer.initializer(init, shape=None, dtype=mstype.float32)

    创建并初始化一个Tensor。

    **参数：**

    - **init** (Union[Tensor, str, Initializer, numbers.Number]) – 初始化方式。

      - **str** - `init` 是继承自 `Initializer` 的类的别名，实际使用时会调用相应的类。`init` 的值可以是"normal"、"ones"或"zeros"等。
      - **Initializer** - `init` 是继承自 `Initializer` ，用于初始化Tensor的类。
      - **numbers.Number** - 调用常量来初始化张量。

    - **shape** (Union[[tuple, list, int]) - 被初始化的Tensor的shape，默认值为None。
    - **dtype** (mindspore.dtype) – 被初始化的Tensor的数据类型，默认值为 `mindspore.float32` 。

    **返回：**

    Tensor，返回一个张量对象。

    **异常：**

    - **TypeError** - 参数 `init` 的类型不正确。
    - **ValueError** - 通过 `init` 传入的Tensor的shape和作为参数传入的shape不一致。

    **样例：**

    >>> import mindspore
    >>> from mindspore.common.initializer import initializer, One
    >>> tensor = initializer('ones', [1, 2, 3], mindspore.float32)
    >>> tensor = initializer(One(), [1, 2, 3], mindspore.float32)
    >>> tensor = initializer(0, [1, 2, 3], mindspore.float32)
