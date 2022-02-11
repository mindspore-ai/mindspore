mindspore.common.initializer
=============================

初始化神经元参数。

.. py:class:: mindspore.common.initializer.Initializer(**kwargs)

    初始化器的抽象基类。

    **参数：**

    - **kwargs** (dict) – `Initializer` 的关键字参数。

.. py:class:: mindspore.common.initializer.TruncatedNormal(sigma=0.01)

    生成一个服从截断正态（高斯）分布的随机数组用于初始化Tensor。

    **参数：**

    **sigma** (float) - 截断正态分布的标准差，默认值为0.01。

.. py:class:: mindspore.common.initializer.Normal(sigma=0.01, mean=0.0)

    生成一个服从正态分布N(sigma, mean)的随机数组用于初始化Tensor。

    .. math::
        f(x) =  \frac{1} {\sqrt{2*π} * sigma}exp(-\frac{(x - mean)^2} {2*{sigma}^2})

    **参数：**

    - **sigma** (float) - 正态分布的标准差，默认值为0.01。
    - **mean** (float) - 正态分布的均值，默认值为0.0。

.. py:class:: mindspore.common.initializer.Uniform(scale=0.07)

    生成一个服从均匀分布U(-scale, scale)的随机数组用于初始化Tensor。

    **参数：**

    **scale** (float) - 均匀分布的边界，默认值为0.07。

.. py:class:: mindspore.common.initializer.HeUniform(negative_slope=0, mode="fan_in", nonlinearity="leaky_relu")

    生成一个服从HeKaiming均匀分布U(-boundary, boundary)的随机数组用于初始化Tensor，其中：
    
    .. math::
        boundary = \text{gain} \times \sqrt{\frac{3}{fan\_mode}}

    其中，gain是一个可选的缩放因子。fan_mode是权重Tensor中输入或输出单元的数量，取决于mode是"fan_in"或是"fan_out"。

    **参数：**

    - **negative_slope** (int, float, bool) - 本层激活函数的负数区间斜率（仅适用于非线性激活函数"leaky_relu"），默认值为0。
    - **mode** (str) - 可选"fan_in"或"fan_out"，"fan_in"会保留前向传递中权重方差的量级，"fan_out"会保留反向传递的量级，默认为"fan_in"。
    - **nonlinearity** (str) - 非线性激活函数，推荐使用"relu"或"leaky_relu"，默认为"leaky_relu"。

.. py:class:: mindspore.common.initializer.HeNormal(negative_slope=0, mode="fan_in", nonlinearity="leaky_relu")

    生成一个服从HeKaiming正态分布N(0, sigma^2)的随机数组用于初始化Tensor，其中：

    .. math::
        sigma = \frac{gain} {\sqrt{fan\_mode}}

    其中，gain是一个可选的缩放因子。如果mode是"fan_in"，则fan_mode是权重Tensor中输入单元的数量，如果mode是"fan_out"，
    fan_mode是权重Tensor中输出单元的数量。

    HeUniform 算法的详细信息，请查看 https://arxiv.org/abs/1502.01852。

    **参数：**

    - **negative_slope** (int, float, bool) - 本层激活函数的负数区间斜率（仅适用于非线性激活函数"leaky_relu"），默认值为0。
    - **mode** (str) - 可选"fan_in"或"fan_out"，"fan_in"会保留前向传递中权重方差的量级，"fan_out"会保留反向传递的量级，默认为"fan_in"。
    - **nonlinearity** (str) - 非线性激活函数，推荐使用"relu"或"leaky_relu"，默认为"leaky_relu"。

.. py:class:: mindspore.common.initializer.XavierUniform(gain=1)

    生成一个服从Xarvier均匀分布U(-boundary, boundary)的随机数组用于初始化Tensor，均匀分布的取值范围为[-boundary, boundary]，其中：

    .. math::
        boundary = gain * \sqrt{\frac{6}{n_{in} + n_{out}}}

    :math:`gain` 是一个可选的缩放因子。:math:`n_{in}` 为权重Tensor中输入单元的数量。:math:`n_{out}` 为权重Tensor中输出单元的数量。

    有关 XavierUniform 算法的详细信息，请查看 http://proceedings.mlr.press/v9/glorot10a.html。

    **参数：**

    **gain** (float) - 可选的缩放因子，默认值为1。

.. py:class:: mindspore.common.initializer.One(**kwargs)

    生成一个值全为1的常量数组用于初始化Tensor。

.. py:class:: mindspore.common.initializer.Zero(**kwargs)

    生成一个值全为0的常量数组用于初始化Tensor。

.. py:class:: mindspore.common.initializer.Constant(value)

    生成一个常量数组用于初始化Tensor。

    **参数：**

    **value** (Union[int, numpy.ndarray]) - 用于初始化的常数值或者数组。




.. py:function:: mindspore.common.initializer.initializer(init, shape=None, dtype=mstype.float32)

    创建并初始化一个Tensor。

    **参数：**

    - **init** (Union[Tensor, str, Initializer, numbers.Number]) – 初始化方式。

      - **str** - `init` 是继承自 `Initializer` 的类的别名，实际使用时会调用相应的类。`init` 的值可以是"normal"、"ones"或"zeros"等。
      - **Initializer** - `init` 是继承自 `Initializer` ，用于初始化Tensor的类。
      - **numbers.Number** - 用于初始化Tensor的常量。

    - **shape** (Union[[tuple, list, int]) - 被初始化的Tensor的shape，默认值为None。
    - **dtype** (mindspore.dtype) – 被初始化的Tensor的数据类型，默认值为 `mindspore.float32` 。

    **返回：**

    Tensor。

    **异常：**

    - **TypeError** - 参数 `init` 的类型不正确。
    - **ValueError** - 当 `init` 传入Tensor对象时， `init` 的shape与形参 `shape` 内的数值不一致。

.. automodule:: mindspore.common.initializer
    :members:
