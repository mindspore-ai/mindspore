mindspore.nn.CellList
======================

.. py:class:: mindspore.nn.CellList(*args, **kwargs)

    构造Cell列表。

    CellList可以像普通Python列表一样使用，支持'__getitem__'、'__setitem__'、'__delitem__'、'__len__'、'__iter__'及'__iadd__'，但包含的Cell都已正确注册，且对所有Cell方法可见。
    
    **参数：**
    
    **args** (list，可选) - 仅包含Cell子类的列表。

    **支持平台：**
    
    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> conv = nn.Conv2d(100, 20, 3)
    >>> bn = nn.BatchNorm2d(20)
    >>> relu = nn.ReLU()
    >>> cell_ls = nn.CellList([bn])
    >>> cell_ls.insert(0, conv)
    >>> cell_ls.append(relu)
    >>> print(cell_ls)
    CellList<
      (0): Conv2d<input_channels=100, output_channels=20, kernel_size=(3, 3),stride=(1, 1),  pad_mode=same,
      padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
      (1): BatchNorm2d<num_features=20, eps=1e-05, momentum=0.09999999999999998, gamma=Parameter (name=1.gamma,
      shape=(20,), dtype=Float32, requires_grad=True), beta=Parameter (name=1.beta, shape=(20,), dtype=Float32,
      requires_grad=True), moving_mean=Parameter (name=1.moving_mean, shape=(20,), dtype=Float32,
      requires_grad=False), moving_variance=Parameter (name=1.moving_variance, shape=(20,), dtype=Float32,
      requires_grad=False)>
      (2): ReLU<>
      >
    
    .. py:method:: append(cell)

        在列表末尾添加一个Cell。

        **参数：**
        
        **cell** (Cell) - 要添加的Cell。

    .. py:method:: extend(cells)

        将Python iterable中的Cell追加到列表的末尾。

        **参数：**

        **cells** (list) - 要追加的Cell子类列表。

        **异常：**
        
        **TypeError** - cells不是Cell子类列表。
        
    .. py:method:: insert(index, cell)

        在列表中的给定索引之前插入给定的Cell。

        **参数：**

        - **index** (int) - 给定的列表索引。
        - **cell** (Cell) - 要插入的Cell子类。