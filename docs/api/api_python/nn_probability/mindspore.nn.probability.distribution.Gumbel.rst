mindspore.nn.probability.distribution.Gumbel
================================================

.. py:class:: mindspore.nn.probability.distribution.Gumbel(loc, scale, seed=0, dtype=mindspore.float32, name='Gumbel')

    耿贝尔分布（Gumbel distribution）。

    **参数：**

    - **loc** (float, list, numpy.ndarray, Tensor) - 耿贝尔分布的位置。
    - **scale** (float, list, numpy.ndarray, Tensor) - 耿贝尔分布的尺度。
    - **seed** (int) - 采样时使用的种子。如果为None，则使用全局种子。默认值：None。
    - **dtype** (mindspore.dtype) - 分布类型。默认值：mindspore.float32。
    - **name** (str) - 分布的名称。默认值：'Gumbel'。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note:: 
        - `scale` 必须大于零。
        - `dtype` 必须是浮点类型，因为耿贝尔分布是连续的。
        - GPU后端不支持 `kl_loss` 和 `cross_entropy` 。

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.distribution as msd
    >>> from mindspore import Tensor
    >>> class Prob(nn.Cell):
    ...     def __init__(self):
    ...         super(Prob, self).__init__()
    ...         self.gum = msd.Gumbel(np.array([0.0]), np.array([[1.0], [2.0]]), dtype=mindspore.float32)
    ...
    ...     def construct(self, x_):
    ...         return self.gum.prob(x_)
    >>> value = np.array([1.0, 2.0]).astype(np.float32)
    >>> pdf = Prob()
    >>> output = pdf(Tensor(value, dtype=mindspore.float32))
    
    .. py:method:: loc
        :property:

        返回分布位置。
        
    .. py:method:: scale
        :property:

        返回分布尺度。
        