mindspore.set_seed
===================

.. py:class:: mindspore.set_seed(seed)

    设置全局种子。

    .. note::
        - 全局种子可用于numpy.random,mindspore.common.Initializer,mindspore.ops.composite.random_ops以及mindspore.nn.probability.distribution。
        - 如果没有设置全局种子，这些包将会各自使用自己的种子，numpy.random和mindspore.common.Initializer将会随机选择种子值，mindspore.ops.composite.random_ops和mindspore.nn.probability.distribution将会使用零作为种子值。
        - numpy.random.seed()设置的种子仅能被numpy.random使用，而这个API设置的种子也可被numpy.random使用，因此推荐使用这个API设置所有的种子。

    **参数：**

    - **seed** (int) – 设置的全局种子。

    **异常:**

    - **ValueError** – 种子值非法 (小于0)。
    - **TypeError** – 种子值非整型数。

    **样例:**

    >>> import numpy as np
    >>> import mindspore as ms
    >>> import mindspore.ops as ops
    >>> from mindspore import Tensor, set_seed, Parameter
    >>> from mindspore.common.initializer import initializer
    ...
    >>> # 注意：（1）请确保代码在动态图模式下运行；
    >>> # （2）由于复合级别的算子需要参数为张量类型，如以下样例，
    >>> # 当使用ops.uniform这个算子，minval和maxval用以下方法初始化:
    >>> minval = Tensor(1.0, ms.float32)
    >>> maxval = Tensor(2.0, ms.float32)
    ...
    >>> # 1. 如果没有设置全局种子，numpy.random以及initializer将会选择随机种子：
    >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A1
    >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A2
    >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W1
    >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W2
    >>> # 重新运行程序将得到不同的结果：
    >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A3
    >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A4
    >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W3
    >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W4
    ...
    >>> # 2. 如果设置了全局种子，numpy.random以及initializer将会使用这个种子：
    >>> set_seed(1234)
    >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A1
    >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A2
    >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W1
    >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W2
    >>> # 重新运行程序将得到相同的结果：
    >>> set_seed(1234)
    >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A1
    >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A2
    >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W1
    >>> w1 = Parameter(initializer("uniform", [2, 2], ms.float32), name="w1") # W2
    ...
    >>> # 3. 如果全局种子或者算子种子均未设置，mindspore.ops.composite.random_ops以及
    >>> # mindspore.nn.probability.distribution将会选择一个随机种子：
    >>> c1 = ops.uniform((1, 4), minval, maxval) # C1
    >>> c2 = ops.uniform((1, 4), minval, maxval) # C2
    >>> # 重新运行程序将得到不同的结果：
    >>> c1 = ops.uniform((1, 4), minval, maxval) # C3
    >>> c2 = ops.uniform((1, 4), minval, maxval) # C4
    ...
    >>> # 4. 如果设置了全局种子，但未设置算子种子，mindspore.ops.composite.random_ops以及
    >>> # mindspore.nn.probability.distribution将会根据全局种子及默认算子种子计算出一个种子。
    >>> # 每次调用默认算子种子都会改变，因此每次调用会得到不同的结果。
    >>> set_seed(1234)
    >>> c1 = ops.uniform((1, 4), minval, maxval) # C1
    >>> c2 = ops.uniform((1, 4), minval, maxval) # C2
    >>> # 重新运行程序将得到相同的结果：
    >>> set_seed(1234)
    >>> c1 = ops.uniform((1, 4), minval, maxval) # C1
    >>> c2 = ops.uniform((1, 4), minval, maxval) # C2
    ...
    >>> # 5. 如果设置了全局种子以及算子种子，mindspore.ops.composite.random_ops以及
    >>> # mindspore.nn.probability.distribution将根据全局种子及算子种子计数器计算出一个种子。
    >>> # 每次调用将会更改算子种子计数器, 因此每次调用会得到不同的结果。
    >>> set_seed(1234)
    >>> c1 = ops.uniform((1, 4), minval, maxval, seed=2) # C1
    >>> c2 = ops.uniform((1, 4), minval, maxval, seed=2) # C2
    >>> # 重新运行程序将得到相同的结果：
    >>> set_seed(1234)
    >>> c1 = ops.uniform((1, 4), minval, maxval, seed=2) # C1
    >>> c2 = ops.uniform((1, 4), minval, maxval, seed=2) # C2
    ...
    >>> # 6. 如果算子种子设置了但是全局种子没有设置，0将作为全局种子，那么
    >>> # mindspore.ops.composite.random_ops以及mindspore.nn.probability.distribution运行方式同5。
    >>> c1 = ops.uniform((1, 4), minval, maxval, seed=2) # C1
    >>> c2 = ops.uniform((1, 4), minval, maxval, seed=2) # C2
    >>> # 重新运行程序将得到相同的结果：
    >>> c1 = ops.uniform((1, 4), minval, maxval, seed=2) # C1
    >>> c2 = ops.uniform((1, 4), minval, maxval, seed=2) # C2
    ...
    >>> # 7. 在程序中重新调用set_seed()将会重置mindspore.ops.composite.random_ops
    >>> # 和mindspore.nn.probability.distribution的numpy种子以及算子种子计数器。
    >>> set_seed(1234)
    >>> np_1 = np.random.normal(0, 1, [1]).astype(np.float32) # A1
    >>> c1 = ops.uniform((1, 4), minval, maxval, seed=2) # C1
    >>> set_seed(1234)
    >>> np_2 = np.random.normal(0, 1, [1]).astype(np.float32) # still get A1
    >>> c2 = ops.uniform((1, 4), minval, maxval, seed=2) # still get C1
