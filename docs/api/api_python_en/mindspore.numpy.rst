mindspore.numpy
===============

.. currentmodule:: mindspore.numpy

MindSpore Numpy package contains a set of Numpy-like interfaces, which allows developers to build models on MindSpore with similar syntax of Numpy.

MindSpore Numpy operators can be classified into four functional modules: `array generation`, `array operation`, `logic operation` and `math operation`.

Common imported modules in corresponding API examples are as follows:

.. code-block:: python

    import mindspore.numpy as np

.. note::
    
    MindSpore numpy provides a consistent programming experience with native numpy by assembling the low-level operators. Compared with MindSpore's function and ops interfaces, it is easier for user to understand and use. However, please notice that to be more compatible with native numpy, the performance of some MindSpore numpy interfaces may be weaker than the corresponding function/ops interfaces. Users can choose which to use as needed.

Array Generation
----------------

Array generation operators are used to generate tensors.

Here is an example to generate an array:

.. code-block:: python

    import mindspore.numpy as np
    import mindspore.ops as ops

    input_x = np.array([1, 2, 3], np.float32)
    print("input_x =", input_x)
    print("type of input_x =", ops.typeof(input_x))

The result is as follows:

.. code-block::

    input_x = [1. 2. 3.]
    type of input_x = Tensor[Float32]
    
Here we have more examples:

- Generate a tensor filled with the same element

  `np.full` can be used to generate a tensor with user-specified values:

  .. code-block:: python

      input_x = np.full((2, 3), 6, np.float32)
      print(input_x)

  The result is as follows:

  .. code-block::

      [[6. 6. 6.]
       [6. 6. 6.]]
    

  Here is another example to generate an array with the specified shape and filled with the value of 1:

  .. code-block:: python

      input_x = np.ones((2, 3), np.float32)
      print(input_x)

  The result is as follows:

  .. code-block::

      [[1. 1. 1.]
       [1. 1. 1.]]
    

- Generate tensors in a specified range

  Generate an arithmetic array within the specified range：

  .. code-block:: python

      input_x = np.arange(0, 5, 1)
      print(input_x)

  The result is as follows:

  .. code-block::

      [0 1 2 3 4]
    

- Generate tensors with specific requirement

  Generate a matrix where the lower elements are 1 and the upper elements are 0 on the given diagonal:

  .. code-block:: python

      input_x = np.tri(3, 3, 1)
      print(input_x)

  The result is as follows:

  .. code-block::

      [[1. 1. 0.]
       [1. 1. 1.]
       [1. 1. 1.]]
    

  Another example, generate a 2-D matrix with a diagonal of 1 and other elements of 0:

  .. code-block:: python

      input_x = np.eye(2, 2)
      print(input_x)

  The result is as follows:

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

Array Operation
---------------

Array operations focus on tensor manipulation.

- Manipulate the shape of the tensor

  For example, transpose a matrix:

  .. code-block:: python

      input_x = np.arange(10).reshape(5, 2)
      output = np.transpose(input_x)
      print(output)

  The result is as follows:

  .. code-block::

      [[0 2 4 6 8]
       [1 3 5 7 9]]
    

  Another example, swap two axes:

  .. code-block:: python

      input_x = np.ones((1, 2, 3))
      output = np.swapaxes(input_x, 0, 1)
      print(output.shape)

  The result is as follows:

  .. code-block::

      (2, 1, 3)
    

- Tensor splitting

  Divide the input tensor into multiple tensors equally, for example:

  .. code-block:: python

      input_x = np.arange(9)
      output = np.split(input_x, 3)
      print(output)

  The result is as follows:

  .. code-block::

      (Tensor(shape=[3], dtype=Int32, value= [0, 1, 2]), Tensor(shape=[3], dtype=Int32, value= [3, 4, 5]), Tensor(shape=[3], dtype=Int32, value= [6, 7, 8]))

- Tensor combination

  Concatenate the two tensors according to the specified axis, for example:

  .. code-block:: python

      input_x = np.arange(0, 5)
      input_y = np.arange(10, 15)
      output = np.concatenate((input_x, input_y), axis=0)
      print(output)

  The result is as follows:

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

Logic
-----

Logic operations define computations related with boolean types.
Examples of `equal` and `less` operations are as follows:

.. code-block:: python

    input_x = np.arange(0, 5)
    input_y = np.arange(0, 10, 2)
    output = np.equal(input_x, input_y)
    print("output of equal:", output)
    output = np.less(input_x, input_y)
    print("output of less:", output)

The result is as follows:

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

Math
----

Math operations include basic and advanced math operations on tensors, and they have full support on Numpy broadcasting rules. Here are some examples:

- Sum two tensors

  The following code implements the operation of adding two tensors of `input_x` and `input_y`:

  .. code-block:: python

      input_x = np.full((3, 2), [1, 2])
      input_y = np.full((3, 2), [3, 4])
      output = np.add(input_x, input_y)
      print(output)

  The result is as follows:

  .. code-block::

      [[4 6]
       [4 6]
       [4 6]]

- Matrics multiplication

  The following code implements the operation of multiplying two matrices `input_x` and `input_y`:

  .. code-block:: python

      input_x = np.arange(2*3).reshape(2, 3).astype('float32')
      input_y = np.arange(3*4).reshape(3, 4).astype('float32')
      output = np.matmul(input_x, input_y)
      print(output)

  The result is as follows:

  .. code-block::

      [[20. 23. 26. 29.]
       [56. 68. 80. 92.]]
    

- Take the average along a given axis

  The following code implements the operation of averaging all the elements of `input_x`:

  .. code-block:: python

      input_x = np.arange(6).astype('float32')
      output = np.mean(input_x)
      print(output)

  The result is as follows:

  .. code-block::

      2.5
    

- Exponential arithmetic

  The following code implements the operation of the natural constant `e` to the power of `input_x`:

  .. code-block:: python

      input_x = np.arange(5).astype('float32')
      output = np.exp(input_x)
      print(output)

  The result is as follows:

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

Interact With MindSpore Functions
---------------------------------

Since `mindspore.numpy` directly wraps MindSpore tensors and operators, it has all the advantages and properties of MindSpore. In this section, we will briefly introduce how to employ MindSpore execution management and automatic differentiation in `mindspore.numpy` coding scenarios. These include:

- `ms_function`: for running codes in static graph mode for better efficiency.
- `GradOperation`: for automatic gradient computation.
- `mindspore.set_context`: for `mindspore.numpy` execution management.
- `mindspore.nn.Cell`: for using `mindspore.numpy` interfaces in MindSpore Deep Learning Models.

The following are examples:

- Use ms_function to run code in static graph mode

  Let's first see an example consisted of matrix multiplication and bias add, which is a typical process in Neural Networks:

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

  The result is as follows:

  .. code-block::

      [[ 768.  768.  768.  768.]
       [2816. 2816. 2816. 2816.]]
    

  In this function, MindSpore dispatches each computing kernel to device separately. However, with the help of `ms_function`, we can compile all operations into a single static computing graph.

  .. code-block:: python

      from mindspore import ms_function

      forward_compiled = ms_function(forward)
      print(forward(x, w1, b1, w2, b2, w3, b3))

  The result is as follows:

  .. code-block::

      [[ 768.  768.  768.  768.]
       [2816. 2816. 2816. 2816.]]
    
  .. note::
      Currently, static graph cannot run in Python interactive mode and not all python types can be passed into functions decorated with `ms_function`. For details about how to use `ms_function`, see `API ms_function <https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.ms_function.html>`_ .

- Use GradOperation to compute deratives

  `GradOperation` can be used to take deratives from normal functions and functions decorated with `ms_function`. Take the previous example:

  .. code-block:: python

      from mindspore import ops

      grad_all = ops.composite.GradOperation(get_all=True)
      print(grad_all(forward)(x, w1, b1, w2, b2, w3, b3))

  The result is as follows:

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

  To take the gradient of `ms_function` compiled functions, first we need to set the execution mode to static graph mode.


  .. code-block:: python

      from mindspore import ms_function, set_context, GRAPH_MODE

      set_context(mode=GRAPH_MODE)
      grad_all = ops.composite.GradOperation(get_all=True)
      print(grad_all(ms_function(forward))(x, w1, b1, w2, b2, w3, b3))

  The result is as follows:

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

  For more details, see `API GradOperation <https://www.mindspore.cn/docs/en/r1.9/api_python/ops/mindspore.ops.GradOperation.html>`_ .

- Use mindspore.set_context to control execution mode

  Most functions in `mindspore.numpy` can run in Graph Mode and PyNative Mode, and can run on CPU, GPU and Ascend. Like MindSpore, users can manage the execution mode using `mindspore.set_context`：

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

  For more details, see `API mindspore.set_context <https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore/mindspore.set_context.html#mindspore.set_context>`_ .

- Use mindspore.numpy in MindSpore Deep Learning Models

  `mindspore.numpy` interfaces can be used inside `nn.cell` blocks as well. For example, the above code can be modified to:

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

  The result is as follows:

  .. code-block::

      [[ 768.  768.  768.  768.]
       [2816. 2816. 2816. 2816.]]