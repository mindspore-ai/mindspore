mindspore.nn.LeakyReLU
=======================

.. py:class:: mindspore.nn.LeakyReLU(alpha=0.2)

   Leaky ReLU激活函数。

   LeakyReLU与ReLU相似，但LeakyReLU有一个斜率，使其在x<0时不等于0，该激活函数定义如下：

   .. math::
      \text{leaky_relu}(x) = \begin{cases}x, &\text{if } x \geq 0; \cr
      \text{alpha} * x, &\text{otherwise.}\end{cases}


   更多细节详见 `Rectifier Nonlinearities Improve Neural Network Acoustic Models <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`_。

   **参数：**

   **alpha** (`Union[int, float]`) – x<0时激活函数的斜率，默认值：0.2。

   **输入：**

   **x** （Tensor） - LeakyReLU的输入。shape为 :math:`(N, *)` ，其中 :math:`*` 表示任意的附加维度数。

   **输出：**

   Tensor，shape和数据类型与 `x` 的相同。

   **异常：**

   **TypeError** - `alpha` 不是浮点数或整数。

   **支持平台：**

   ``Ascend`` ``GPU`` ``CPU``

   **样例：**

	>>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
	>>> leaky_relu = nn.LeakyReLU()
	>>> output = leaky_relu(x)
	>>> print(output)
	[[-0.2  4.  -1.6]
	 [ 2.  -1.   9. ]]