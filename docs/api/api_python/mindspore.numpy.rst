mindspore.numpy
===============
	
.. currentmodule:: mindspore.numpy
	
MindSpore NumPy工具包提供了一系列类NumPy接口。用户可以使用类NumPy语法在MindSpore上进行模型的搭建。

MindSpore Numpy具有四大功能模块：Array生成、Array操作、逻辑运算和数学运算。

在API示例中，常用的模块导入方法如下：

.. code-block::

    import mindspore.numpy as np

.. note::

    MindSpore numpy通过组装底层算子来提供与numpy一致的编程体验接口，方便开发人员使用和代码移植。相比于MindSpore的function和ops接口，与原始numpy的接口格式及行为一致性更好，以便于用户理解和使用。注意：由于兼容numpy的考虑，部分接口的性能可能弱于function和ops接口。使用者可以按需选择不同类型的接口。

Array生成
----------------

生成类算子用来生成和构建具有指定数值、类型和形状的数组(Tensor)。

构建数组代码示例：

.. code-block:: python

    import mindspore.numpy as np
    import mindspore.ops as ops
    input_x = np.array([1, 2, 3], np.float32)
    print("input_x =", input_x)
    print("type of input_x =", ops.typeof(input_x))
	
运行结果如下：

.. code-block::

    input_x = [1. 2. 3.]
    type of input_x = Tensor[Float32]

除了使用上述方法来创建外，也可以通过以下几种方式创建。

- 生成具有相同元素的数组

  生成具有相同元素的数组代码示例：

  .. code-block:: python

      input_x = np.full((2, 3), 6, np.float32)
      print(input_x)

  运行结果如下：

  .. code-block::

      [[6. 6. 6.]
       [6. 6. 6.]]
    
  生成指定形状的全1数组，示例：

  .. code-block:: python

      input_x = np.ones((2, 3), np.float32)
      print(input_x)

  运行结果如下：

  .. code-block::

      [[1. 1. 1.]
       [1. 1. 1.]]
    
- 生成具有某个范围内的数值的数组

  生成指定范围内的等差数组代码示例：

  .. code-block:: python

      input_x = np.arange(0, 5, 1)
      print(input_x)

  运行结果如下：

  .. code-block::

      [0 1 2 3 4]
    

- 生成特殊类型的数组

  生成给定对角线处下方元素为1，上方元素为0的矩阵，示例：

  .. code-block:: python

      input_x = np.tri(3, 3, 1)
      print(input_x)

  运行结果如下：

  .. code-block::

      [[1. 1. 0.]
       [1. 1. 1.]
       [1. 1. 1.]]
    
  生成对角线为1，其他元素为0的二维矩阵，示例：

  .. code-block:: python

      input_x = np.eye(2, 2)
      print(input_x)

  运行结果如下：

  .. code-block::

      [[1. 0.]
       [0. 1.]] 

.. msplatformautosummary::
    :toctree: numpy
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.numpy.arange
    mindspore.numpy.array
    mindspore.numpy.asarray
    mindspore.numpy.asfarray
    mindspore.numpy.bartlett
    mindspore.numpy.blackman
    mindspore.numpy.copy
    mindspore.numpy.diag
    mindspore.numpy.diag_indices
    mindspore.numpy.diagflat
    mindspore.numpy.diagonal
    mindspore.numpy.empty
    mindspore.numpy.empty_like
    mindspore.numpy.eye
    mindspore.numpy.full
    mindspore.numpy.full_like
    mindspore.numpy.geomspace
    mindspore.numpy.hamming
    mindspore.numpy.hanning
    mindspore.numpy.histogram_bin_edges
    mindspore.numpy.identity
    mindspore.numpy.indices
    mindspore.numpy.ix_
    mindspore.numpy.linspace
    mindspore.numpy.logspace
    mindspore.numpy.meshgrid
    mindspore.numpy.mgrid
    mindspore.numpy.ogrid
    mindspore.numpy.ones
    mindspore.numpy.ones_like
    mindspore.numpy.pad
    mindspore.numpy.rand
    mindspore.numpy.randint
    mindspore.numpy.randn
    mindspore.numpy.trace
    mindspore.numpy.tri
    mindspore.numpy.tril
    mindspore.numpy.tril_indices
    mindspore.numpy.tril_indices_from
    mindspore.numpy.triu
    mindspore.numpy.triu_indices
    mindspore.numpy.triu_indices_from
    mindspore.numpy.vander
    mindspore.numpy.zeros
    mindspore.numpy.zeros_like

Array操作
---------------

操作类算子主要进行数组的维度变换，分割和拼接等。

- 数组维度变换

  矩阵转置，代码示例：

  .. code-block:: python

      input_x = np.arange(10).reshape(5, 2)
      output = np.transpose(input_x)
      print(output)

  运行结果如下：

  .. code-block::

      [[0 2 4 6 8]
       [1 3 5 7 9]]
    
  交换指定轴，代码示例：

  .. code-block:: python

      input_x = np.ones((1, 2, 3))
      output = np.swapaxes(input_x, 0, 1)
      print(output.shape)

  运行结果如下：

  .. code-block::

      (2, 1, 3)
    

- 数组分割

  将输入数组平均切分为多个数组，代码示例：

  .. code-block:: python

      input_x = np.arange(9)
      output = np.split(input_x, 3)
      print(output)

  运行结果如下：

  .. code-block::

      (Tensor(shape=[3], dtype=Int32, value= [0, 1, 2]), Tensor(shape=[3], dtype=Int32, value= [3, 4, 5]), Tensor(shape=[3], dtype=Int32, value= [6, 7, 8])) 

- 数组拼接

  将两个数组按照指定轴进行拼接，代码示例：

  .. code-block:: python

      input_x = np.arange(0, 5)
      input_y = np.arange(10, 15)
      output = np.concatenate((input_x, input_y), axis=0)
      print(output)

  运行结果如下：

  .. code-block::

      [ 0  1  2  3  4 10 11 12 13 14]
   

.. msplatformautosummary::
    :toctree: numpy
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.numpy.append
    mindspore.numpy.apply_along_axis
    mindspore.numpy.apply_over_axes
    mindspore.numpy.array_split
    mindspore.numpy.array_str
    mindspore.numpy.atleast_1d
    mindspore.numpy.atleast_2d
    mindspore.numpy.atleast_3d
    mindspore.numpy.broadcast_arrays
    mindspore.numpy.broadcast_to
    mindspore.numpy.choose
    mindspore.numpy.column_stack
    mindspore.numpy.concatenate
    mindspore.numpy.dsplit
    mindspore.numpy.dstack
    mindspore.numpy.expand_dims
    mindspore.numpy.flip
    mindspore.numpy.fliplr
    mindspore.numpy.flipud
    mindspore.numpy.hsplit
    mindspore.numpy.hstack
    mindspore.numpy.moveaxis
    mindspore.numpy.piecewise
    mindspore.numpy.ravel
    mindspore.numpy.repeat
    mindspore.numpy.reshape
    mindspore.numpy.roll
    mindspore.numpy.rollaxis
    mindspore.numpy.rot90
    mindspore.numpy.select
    mindspore.numpy.size
    mindspore.numpy.split
    mindspore.numpy.squeeze
    mindspore.numpy.stack
    mindspore.numpy.swapaxes
    mindspore.numpy.take
    mindspore.numpy.take_along_axis
    mindspore.numpy.tile
    mindspore.numpy.transpose
    mindspore.numpy.unique
    mindspore.numpy.unravel_index
    mindspore.numpy.vsplit
    mindspore.numpy.vstack
    mindspore.numpy.where

逻辑运算
-----------

逻辑运算类算子主要进行各类逻辑相关的运算。

相等（equal）和小于（less）计算代码示例如下：

.. code-block:: python

    input_x = np.arange(0, 5)
    input_y = np.arange(0, 10, 2)
    output = np.equal(input_x, input_y)
    print("output of equal:", output)
    output = np.less(input_x, input_y)
    print("output of less:", output)

运行结果如下：

.. code-block::

    output of equal: [ True False False False False]
    output of less: [False  True  True  True  True]
  
.. msplatformautosummary::
    :toctree: numpy
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.numpy.array_equal
    mindspore.numpy.array_equiv
    mindspore.numpy.equal
    mindspore.numpy.greater
    mindspore.numpy.greater_equal
    mindspore.numpy.in1d
    mindspore.numpy.isclose
    mindspore.numpy.isfinite
    mindspore.numpy.isin
    mindspore.numpy.isinf
    mindspore.numpy.isnan
    mindspore.numpy.isneginf
    mindspore.numpy.isposinf
    mindspore.numpy.isscalar
    mindspore.numpy.less
    mindspore.numpy.less_equal
    mindspore.numpy.logical_and
    mindspore.numpy.logical_not
    mindspore.numpy.logical_or
    mindspore.numpy.logical_xor
    mindspore.numpy.not_equal
    mindspore.numpy.signbit
    mindspore.numpy.sometrue

数学运算
-------------

数学运算类算子包括各类数学相关的运算：加减乘除乘方，以及指数、对数等常见函数等。

数学计算支持类似NumPy的广播特性。

- 加法

  以下代码实现了 `input_x` 和 `input_y` 两数组相加的操作：

  .. code-block:: python

      input_x = np.full((3, 2), [1, 2])
      input_y = np.full((3, 2), [3, 4])
      output = np.add(input_x, input_y)
      print(output)

  运行结果如下：

  .. code-block::

      [[4 6]
       [4 6]
       [4 6]]

- 矩阵乘法

  以下代码实现了 `input_x` 和 `input_y` 两矩阵相乘的操作：

  .. code-block:: python

      input_x = np.arange(2*3).reshape(2, 3).astype('float32')
      input_y = np.arange(3*4).reshape(3, 4).astype('float32')
      output = np.matmul(input_x, input_y)
      print(output)

  运行结果如下：

  .. code-block::

      [[20. 23. 26. 29.]
       [56. 68. 80. 92.]] 

- 求平均值

  以下代码实现了求 `input_x` 所有元素的平均值的操作：

  .. code-block:: python

      input_x = np.arange(6).astype('float32')
      output = np.mean(input_x)
      print(output)

  运行结果如下：

  .. code-block::

      2.5  

- 指数

  以下代码实现了自然常数 `e` 的 `input_x` 次方的操作：

  .. code-block:: python

      input_x = np.arange(5).astype('float32')
      output = np.exp(input_x)
      print(output)

  运行结果如下：

  .. code-block::

      [ 1.         2.7182817  7.389056  20.085537  54.59815  ]

.. msplatformautosummary::
    :toctree: numpy
    :nosignatures:
    :template: classtemplate_inherited.rst

    mindspore.numpy.absolute
    mindspore.numpy.add
    mindspore.numpy.amax
    mindspore.numpy.amin
    mindspore.numpy.arccos
    mindspore.numpy.arccosh
    mindspore.numpy.arcsin
    mindspore.numpy.arcsinh
    mindspore.numpy.arctan
    mindspore.numpy.arctan2
    mindspore.numpy.arctanh
    mindspore.numpy.argmax
    mindspore.numpy.argmin
    mindspore.numpy.around
    mindspore.numpy.average
    mindspore.numpy.bincount
    mindspore.numpy.bitwise_and
    mindspore.numpy.bitwise_or
    mindspore.numpy.bitwise_xor
    mindspore.numpy.cbrt
    mindspore.numpy.ceil
    mindspore.numpy.clip
    mindspore.numpy.convolve
    mindspore.numpy.copysign
    mindspore.numpy.corrcoef
    mindspore.numpy.correlate
    mindspore.numpy.cos
    mindspore.numpy.cosh
    mindspore.numpy.count_nonzero
    mindspore.numpy.cov
    mindspore.numpy.cross
    mindspore.numpy.cumprod
    mindspore.numpy.cumsum
    mindspore.numpy.deg2rad
    mindspore.numpy.diff
    mindspore.numpy.digitize
    mindspore.numpy.divide
    mindspore.numpy.divmod
    mindspore.numpy.dot
    mindspore.numpy.ediff1d
    mindspore.numpy.exp
    mindspore.numpy.exp2
    mindspore.numpy.expm1
    mindspore.numpy.fix
    mindspore.numpy.float_power
    mindspore.numpy.floor
    mindspore.numpy.floor_divide
    mindspore.numpy.fmod
    mindspore.numpy.gcd
    mindspore.numpy.gradient
    mindspore.numpy.heaviside
    mindspore.numpy.histogram
    mindspore.numpy.histogram2d
    mindspore.numpy.histogramdd
    mindspore.numpy.hypot
    mindspore.numpy.inner
    mindspore.numpy.interp
    mindspore.numpy.invert
    mindspore.numpy.kron
    mindspore.numpy.lcm
    mindspore.numpy.log
    mindspore.numpy.log10
    mindspore.numpy.log1p
    mindspore.numpy.log2
    mindspore.numpy.logaddexp
    mindspore.numpy.logaddexp2
    mindspore.numpy.matmul
    mindspore.numpy.matrix_power
    mindspore.numpy.maximum
    mindspore.numpy.mean
    mindspore.numpy.minimum
    mindspore.numpy.multi_dot
    mindspore.numpy.multiply
    mindspore.numpy.nancumsum
    mindspore.numpy.nanmax
    mindspore.numpy.nanmean
    mindspore.numpy.nanmin
    mindspore.numpy.nanstd
    mindspore.numpy.nansum
    mindspore.numpy.nanvar
    mindspore.numpy.negative
    mindspore.numpy.norm
    mindspore.numpy.outer
    mindspore.numpy.polyadd
    mindspore.numpy.polyder
    mindspore.numpy.polyint
    mindspore.numpy.polymul
    mindspore.numpy.polysub
    mindspore.numpy.polyval
    mindspore.numpy.positive
    mindspore.numpy.power
    mindspore.numpy.promote_types
    mindspore.numpy.ptp
    mindspore.numpy.rad2deg
    mindspore.numpy.radians
    mindspore.numpy.ravel_multi_index
    mindspore.numpy.reciprocal
    mindspore.numpy.remainder
    mindspore.numpy.result_type
    mindspore.numpy.rint
    mindspore.numpy.searchsorted
    mindspore.numpy.sign
    mindspore.numpy.sin
    mindspore.numpy.sinh
    mindspore.numpy.sqrt
    mindspore.numpy.square
    mindspore.numpy.std
    mindspore.numpy.subtract
    mindspore.numpy.sum
    mindspore.numpy.tan
    mindspore.numpy.tanh
    mindspore.numpy.tensordot
    mindspore.numpy.trapz
    mindspore.numpy.true_divide
    mindspore.numpy.trunc
    mindspore.numpy.unwrap
    mindspore.numpy.var

MindSpore Numpy与MindSpore特性结合
-----------------------------------------

mindspore.numpy能够充分利用MindSpore的强大功能，实现算子的自动微分，并使用图模式加速运算，帮助用户快速构建高效的模型。同时，MindSpore还支持多种后端设备，包括Ascend、GPU和CPU等，用户可以根据自己的需求灵活设置。以下提供了几种常用方法：

- `ms_function`: 将代码包裹进图模式，用于提高代码运行效率。
- `GradOperation`: 用于自动求导。
- `mindspore.set_context`: 用于设置运行模式和后端设备等。
- `mindspore.nn.Cell`: 用于建立深度学习模型。

使用示例如下：

- ms_function使用示例

  首先，以神经网络里经常使用到的矩阵乘与矩阵加算子为例：

  .. code-block:: python

      import mindspore.numpy as np

      x = np.arange(8).reshape(2, 4).astype('float32')
      w1 = np.ones((4, 8))
      b1 = np.zeros((8,))
      w2 = np.ones((8, 16))
      b2 = np.zeros((16,))
      w3 = np.ones((16, 4))
      b3 = np.zeros((4,))

      def forward(x, w1, b1, w2, b2, w3, b3):
          x = np.dot(x, w1) + b1
          x = np.dot(x, w2) + b2
          x = np.dot(x, w3) + b3
          return x

      print(forward(x, w1, b1, w2, b2, w3, b3))

  运行结果如下：

  .. code-block::

      [[ 768.  768.  768.  768.]
       [2816. 2816. 2816. 2816.]]
    

  对上述示例，我们可以借助 `ms_function` 将所有算子编译到一张静态图里以加快运行效率，示例如下：

  .. code-block:: python

      from mindspore import ms_function

      forward_compiled = ms_function(forward)
      print(forward(x, w1, b1, w2, b2, w3, b3))

  运行结果如下：

  .. code-block::

      [[ 768.  768.  768.  768.]
       [2816. 2816. 2816. 2816.]]
  
  .. note::
      目前静态图不支持在Python交互式模式下运行，并且有部分语法限制。`ms_function` 的更多信息可参考 `API ms_function <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/mindspore.ms_function.html>`_ 。

- GradOperation使用示例

  `GradOperation` 可以实现自动求导。以下示例可以实现对上述没有用 `ms_function` 修饰的 `forward` 函数定义的计算求导。

  .. code-block:: python

      from mindspore import ops

      grad_all = ops.composite.GradOperation(get_all=True)
      print(grad_all(forward)(x, w1, b1, w2, b2, w3, b3))

  运行结果如下：

  .. code-block::

      (Tensor(shape=[2, 4], dtype=Float32, value=
       [[ 5.12000000e+02,  5.12000000e+02,  5.12000000e+02,  5.12000000e+02],
        [ 5.12000000e+02,  5.12000000e+02,  5.12000000e+02,  5.12000000e+02]]),
       Tensor(shape=[4, 8], dtype=Float32, value=
       [[ 2.56000000e+02,  2.56000000e+02,  2.56000000e+02 ...  2.56000000e+02,  2.56000000e+02,  2.56000000e+02],
        [ 3.84000000e+02,  3.84000000e+02,  3.84000000e+02 ...  3.84000000e+02,  3.84000000e+02,  3.84000000e+02],
        [ 5.12000000e+02,  5.12000000e+02,  5.12000000e+02 ...  5.12000000e+02,  5.12000000e+02,  5.12000000e+02]
        [ 6.40000000e+02,  6.40000000e+02,  6.40000000e+02 ...  6.40000000e+02,  6.40000000e+02,  6.40000000e+02]]),
        ...
       Tensor(shape=[4], dtype=Float32, value= [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00,  2.00000000e+00]))

  如果要对 `ms_function` 修饰的 `forward` 计算求导，需要提前使用 `set_context` 设置运算模式为图模式，示例如下：

  .. code-block:: python

      from mindspore import ms_function, set_context, GRAPH_MODE

      set_context(mode=GRAPH_MODE)
      grad_all = ops.composite.GradOperation(get_all=True)
      print(grad_all(ms_function(forward))(x, w1, b1, w2, b2, w3, b3))

  运行结果如下：

  .. code-block::

      (Tensor(shape=[2, 4], dtype=Float32, value=
       [[ 5.12000000e+02,  5.12000000e+02,  5.12000000e+02,  5.12000000e+02],
        [ 5.12000000e+02,  5.12000000e+02,  5.12000000e+02,  5.12000000e+02]]),
       Tensor(shape=[4, 8], dtype=Float32, value=
       [[ 2.56000000e+02,  2.56000000e+02,  2.56000000e+02 ...  2.56000000e+02,  2.56000000e+02,  2.56000000e+02],
        [ 3.84000000e+02,  3.84000000e+02,  3.84000000e+02 ...  3.84000000e+02,  3.84000000e+02,  3.84000000e+02],
        [ 5.12000000e+02,  5.12000000e+02,  5.12000000e+02 ...  5.12000000e+02,  5.12000000e+02,  5.12000000e+02]
        [ 6.40000000e+02,  6.40000000e+02,  6.40000000e+02 ...  6.40000000e+02,  6.40000000e+02,  6.40000000e+02]]),
        ...
       Tensor(shape=[4], dtype=Float32, value= [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00,  2.00000000e+00]))

  更多细节可参考 `API GradOperation <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/ops/mindspore.ops.GradOperation.html>`_ 。

- mindspore.set_context使用示例

  MindSpore支持多后端运算，可以通过 `mindspore.set_context` 进行设置。`mindspore.numpy` 的多数算子可以使用图模式或者PyNative模式运行，也可以运行在CPU，CPU或者Ascend等多种后端设备上。

  .. code-block:: python

      from mindspore import set_context, GRAPH_MODE, PYNATIVE_MODE

      # Execucation in static graph mode
      set_context(mode=GRAPH_MODE)

      # Execucation in PyNative mode
      set_context(mode=PYNATIVE_MODE)

      # Execucation on CPU backend
      set_context(device_target="CPU")

      # Execucation on GPU backend
      set_context(device_target="GPU")

      # Execucation on Ascend backend
      set_context(device_target="Ascend")
      ...

  更多细节可参考 `API mindspore.set_context <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/mindspore.set_context.html#mindspore.set_context>`_ 。

- mindspore.numpy使用示例

  这里提供一个使用 `mindspore.numpy` 构建网络模型的示例。

  `mindspore.numpy` 接口可以定义在 `nn.Cell` 代码块内进行网络的构建，示例如下：

  .. code-block:: python

      import mindspore.numpy as np
      from mindspore import set_context, GRAPH_MODE
      from mindspore.nn import Cell

      set_context(mode=GRAPH_MODE)

      x = np.arange(8).reshape(2, 4).astype('float32')
      w1 = np.ones((4, 8))
      b1 = np.zeros((8,))
      w2 = np.ones((8, 16))
      b2 = np.zeros((16,))
      w3 = np.ones((16, 4))
      b3 = np.zeros((4,))

      class NeuralNetwork(Cell):
          def construct(self, x, w1, b1, w2, b2, w3, b3):
              x = np.dot(x, w1) + b1
              x = np.dot(x, w2) + b2
              x = np.dot(x, w3) + b3
              return x

      net = NeuralNetwork()

      print(net(x, w1, b1, w2, b2, w3, b3))

  运行结果如下：

  .. code-block::

      [[ 768.  768.  768.  768.]
       [2816. 2816. 2816. 2816.]]
    