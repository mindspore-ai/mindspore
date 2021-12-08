mindspore.nn.SequentialCell
============================

.. py:class:: mindspore.nn.SequentialCell(*args)

    构造Cell顺序容器。

    Cell列表将按照它们在构造函数中传递的顺序添加到其中。
    或者，也可以传入Cell的有序字典。

    **参数：**

    **args** (list, OrderedDict) - 仅包含Cell子类的列表或有序字典。

    **输入：**

    **x** (Tensor) - Tensor，其shape取决于序列中的第一个Cell。

    **输出：**

    Tensor，输出Tensor，其shape取决于输入 `x` 和定义的Cell序列。

    **异常：**

    **TypeError** - `args` 的类型不是列表或有序字典。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**
    
    >>> conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
    >>> relu = nn.ReLU()
    >>> seq = nn.SequentialCell([conv, relu])
    >>> x = Tensor(np.ones([1, 3, 4, 4]), dtype=mindspore.float32)
    >>> output = seq(x)
    >>> print(output)
    [[[[27. 27.]
       [27. 27.]]
      [[27. 27.]
       [27. 27.]]]]
    

    .. py:method:: append(cell)

        在容器末尾添加一个cell。

        **参数：**
        
        **cell** (Cell) - 要添加的cell。

        **样例：**

        >>> conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
        >>> bn = nn.BatchNorm2d(2)
        >>> relu = nn.ReLU()
        >>> seq = nn.SequentialCell([conv, bn])
        >>> seq.append(relu)
        >>> x = Tensor(np.ones([1, 3, 4, 4]), dtype=mindspore.float32)
        >>> output = seq(x)
        >>> print(output)
        [[[[26.999863 26.999863]
           [26.999863 26.999863]]
          [[26.999863 26.999863]
           [26.999863 26.999863]]]]