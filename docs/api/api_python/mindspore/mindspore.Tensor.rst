mindspore.Tensor
================

.. py:class:: mindspore.Tensor(input_data=None, dtype=None, shape=None, init=None, internal=False)

    张量，即存储多维数组（n-dimensional array）的数据结构。

    **参数：**

    - **input_data** (Union[Tensor, float, int, bool, tuple, list, numpy.ndarray]) - 被存储的数据，可以是其它Tensor，也可以是Python基本数据（如int，float，bool等），或是一个NumPy对象。默认值：None。
    - **dtype** (:class:`mindspore.dtype`) - 用于定义该Tensor的数据类型，必须是 *mindSpore.dtype* 中定义的类型。如果该参数为None，则数据类型与 `input_data` 一致，默认值：None。
    - **shape** (Union[tuple, list, int]) - 用于定义该Tensor的形状。如果指定了 `input_data` ，则无需设置该参数。默认值：None。
    - **init** (Initializer) - 用于在并行模式中延迟Tensor的数据的初始化，如果指定该参数，则 `dtype` 和 `shape` 也必须被指定。不推荐在非自动并行之外的场景下使用该接口。只有当调用 `Tensor.init_data` 时，才会使用指定的 `init` 来初始化Tensor数据。默认值：None。
    - **internal** (bool) - Tensor是否由框架创建。 如果为True，表示Tensor是由框架创建的，如果为False，表示Tensor是由用户创建的。默认值：False。
    
    **输出：**

    Tensor。

    .. py:method:: T
        :property:

        返回转置后的Tensor。

    .. py:method:: abs()

        返回每个元素的绝对值。

        **返回：**

        Tensor。

    .. py:method:: all(axis=(), keep_dims=False)

        检查在指定轴上所有元素是否均为True。

        **参数：**

        - **axis** (Union[None, int, tuple(int)) - 计算all的维度。 当 `axis` 为None或者空元组的时候，计算所有维度。当 `axis` 为int或tuple(int)时，记Tensor的维度为dim，则其取值范围为[-dim, dim)。默认值：()。
        - **keep_dims** (bool) - 计算结果是否保留维度。 默认值： False。

        **返回：**

        Tensor。如果在指定轴方向上所有数组元素都为True，则其值为True，否则其值为False。如果轴为None或空元组，则默认降维。

    .. py:method:: any(axis=(), keep_dims=False)

        检查在指定轴方向上是否存在任意为True的Tensor元素。

        **参数：**

        - **axis** (Union[None, int, tuple(int)) - 计算any的维度。当 `axis` 为None或空元组时，计算所有维度。当 `axis` 为int或tuple(int)时，记Tensor的维度为dim，则其取值范围为[-dim, dim)。默认值：()。
        - **keep_dims** (bool) - 计算结果是否保留维度。默认值：False。

        **返回：**

        Tensor。如果在指定轴方向上所有Tensor元素都为True，则其值为True，否则其值为False。如果轴为None或空元组，则默认降维。

    .. py:method:: argmax(axis=None)

        返回指定轴上最大值的索引。

        **参数：**

        - **axis** (int, optional) - 默认情况下，返回扁平化Tensor的最大值序号，否则返回指定轴方向上。

        **返回：**

        Tensor，最大值的索引。它与原始Tensor具有相同的shape，但移除了轴方向上的维度。

        **异常：**

        - **ValueError** - 入参axis的设定值超出了范围。

    .. py:method:: argmin(axis=None)

        返回指定轴上最小值的索引。

        **参数：**

        - **axis** (int, optional) - 返回扁平化Tensor的最小值序号，否则返回指定轴方向上的最小值序号。默认值: None。

        **返回：**

        Tensor，最小Tensor的索引。它与原始Tensor具有相同的shape，但移除了轴方向上的维度。

        **异常：**

        - **ValueError** - 入参axis的设定值超出了范围。

    .. py:method:: asnumpy()

        将张量转换为NumPy数组。该方法会将Tensor本身转换为NumPy的ndarray。这个Tensor和函数返回的ndarray共享内存地址。对Tensor本身的修改会反映到相应的ndarray上。

        **返回：**

        NumPy的ndarray，该ndarray与Tensor共享内存地址。

    .. py:method:: astype(dtype, copy=True)

        将Tensor转为指定数据类型，可指定是否返回副本。

        **参数：**

        - **dtype** (Union[`mindspore.dtype` , str]) - 指定的Tensor数据类型，可以是: `mindspore.dtype.float32` 或 `float32` 的格式。默认值：`mindspore.dtype.float32` 。
        - **copy** (bool, optional) - 默认情况下，astype返回新拷贝的Tensor。如果该参数设为False，则返回输入Tensor而不是副本。默认值：True。

        **返回：**

        Tensor，指定数据类型的Tensor。

        **异常：**

        - **TypeError** - 指定了无法解析的类型。

    .. py:method:: choose(choices, mode='clip')

        根据原始Tensor数组和一个索引数组构造一个新的Tensor。

        **参数：**

        - **choices** (Union[tuple, list, Tensor]) - 索引选择数组。原始输入Tensor和 `choices` 的广播维度必须相同。如果 `choices` 本身是一个Tensor，则其最外层的维度（即，对应于第0维的维度）被用来定义 `choices` 数组。
        - **mode** ('raise', 'wrap', 'clip', optional) - 指定如何处理 `[0, n-1]` 外部的索引：

          - **raise** – 引发异常（默认）；
          - **wrap** – 原值映射为对n取余后的值；
          - **clip** – 大于n-1的值会被映射为n-1。该模式下禁用负数索引。

        **返回：**

        Tensor，合并后的结果。

        **异常：**

        - **ValueError** - 输入Tensor和任一 `choices` 无法广播。

    .. py:method:: clip(xmin, xmax, dtype=None)

        裁剪Tensor中的值。

        给定一个区间，区间外的值将被裁剪到区间边缘。
        例如，如果指定的间隔为 :math:`[0, 1]` ，则小于0的值将变为0，大于1的值将变为1。

        .. note::
            目前不支持裁剪 `xmin=nan` 或 `xmax=nan` 。

        **参数：**

        - **xmin** (Tensor, scalar, None) - 最小值。如果值为None，则不在间隔的下边缘执行裁剪操作。`xmin` 或 `xmax` 只能有一个为None。
        - **xmax** (Tensor, scalar, None) - 最大值。如果值为None，则不在间隔的上边缘执行裁剪操作。`xmin` 或 `xmax` 只能有一个为None。如果 `xmin` 或 `xmax` 是Tensor，则三个Tensor将被广播进行shape匹配。
        - **dtype** (`mindspore.dtype` , optional) - 覆盖输出Tensor的dtype。默认值为None。

        **返回：**

        Tensor，含有输入Tensor的元素，其中values < `xmin` 被替换为 `xmin` ，values > `xmax` 被替换为 `xmax` 。

        **异常：**

        - **TypeError** - 输入的类型与Tensor不一致。
        - **ValueError** - 输入与Tensor的shape不能广播，或者 `xmin` 和 `xmax` 都是 `None` 。

    .. py:method:: copy()

        复制一个Tensor并返回。

        .. note::
            当前实现不支持类似NumPy的 `order` 参数。

        **返回：**

        复制的Tensor。

    .. py:method:: cumsum(axis=None, dtype=None)

        返回指定轴方向上元素的累加值。

        .. note::
            如果 `dtype` 为 `int8` , `int16` 或 `bool` ，则结果 `dtype` 将提升为 `int32` ，不支持 `int64` 。

        **参数：**

        - **axis** (int, optional) - 轴，在该轴方向上的累积和。默认情况下，计算所有元素的累加和。
        - **dtype** (`mindspore.dtype` , optional) - 如果未指定参数值，则保持与原始Tensor相同，除非参数值是一个精度小于 `float32` 的整数。在这种情况下，使用 `float32` 。默认值：None。

        **异常：**

        - **ValueError** - 轴超出范围。

        **返回：**

        Tensor。

    .. py:method:: diagonal(offset=0, axis1=0, axis2=1)

        返回指定的对角线。

        **参数：**

        - **offset** (int, optional) - 对角线与主对角线的偏移。可以是正值或负值。默认为主对角线。
        - **axis1** (int, optional) - 二维子数组的第一轴，对角线应该从这里开始。默认为第一轴(0)。
        - **axis2** (int, optional) - 二维子数组的第二轴，对角线应该从这里开始。默认为第二轴。

        **返回：**

        Tensor，如果Tensor是二维，则返回值是一维数组。

        **异常：**

        - **ValueError** - 输入Tensor的维度少于2。

    .. py:method:: dtype
        :property:

        返回张量的数据类型（:class:`mindspore.dtype`）。

    .. py:method:: expand_as(x)

        将目标张量的维度扩展为输入张量的维度。

        **参数：**

        - **x** (Tensor) - 输入的张量。

        **返回：**

        维度与输入张量的相同的Tensor。输出张量的维度必须遵守广播规则。广播规则指输出张量的维度需要扩展为输入张量的维度，如果目标张量的维度大于输入张量的维度，则不满足广播规则。

    .. py:method:: expand_dims(axis):

        沿指定轴扩展Tensor维度。

        **参数：**
            - **axis** (int) - 扩展维度指定的轴。

        **返回：**
            Tensor, 指定轴上扩展的维度为1。

        **异常：**
            - **TypeError** - axis不是int类型。
            - **ValueError** - axis的取值不在[-self.ndim - 1, self.ndim + 1)。

    .. py:method:: fill(value)

        用标量值填充数组。

        .. note::
            与NumPy不同，Tensor.fill()将始终返回一个新的Tensor，而不是填充原来的Tensor。

        **参数：**

        - **value** (Union[None, int, float, bool]) - 所有元素都被赋予这个值。

        **返回：**

        Tensor，与原来的dtype和shape相同的Tensor。

        **异常：**

        - **TypeError** - 输入参数具有前面未指定的类型。

    .. py:method:: flatten(order='C')

        返回展开成一维的Tensor的副本。

        **参数：**

        **order** (str, optional) - 可以在'C'和'F'之间进行选择。'C'表示按行优先（C风格）顺序展开。'F'表示按列优先顺序（Fortran风格）进行扁平化。仅支持'C'和'F'。默认值：'C'。

        **返回：**

        Tensor，具有与输入相同的数据类型。

        **异常：**

        - **TypeError** - `order` 不是字符串类型。
        - **ValueError** - `order` 是字符串类型，但不是'C'或'F'。

    .. py:method:: flush_from_cache()

        如果Tensor开启缓存作用，则将缓存数据刷新到host侧。

    .. py:method:: from_numpy(array)
        :staticmethod:

        通过不复制数据的方式将Numpy数组转换为张量。

        **参数：**

        **array** (numpy.array) - 输入数组。

        **返回：**

        与输入的张量具有相同的数据类型的Tensor。

    .. py:method:: has_init
        :property:

        Tensor是否已经初始化。

    .. py:method:: init_data(slice_index=None, shape=None, opt_shard_group=None)

        获取此Tensor的数据。

        .. note:: 对于同一个Tensor，只可以调用一次 `init_data` 函数。

        **参数：**

        - **slice_index** (int) - 参数切片的索引。在初始化参数切片的时候使用，保证使用相同切片的设备可以生成相同的Tensor。默认值：None。
        - **shape** (list[int]) - 切片的shape，在初始化参数切片时使用。默认值：None。
        - **opt_shard_group** (str) - 优化器分片组，在自动或半自动并行模式下用于获取参数的切片。默认值：None。

        **返回：**

        初始化的Tensor。

    .. py:method:: item(index=None)

        获取Tensor中指定索引的元素。

        .. note::
            Tensor.item返回的是Tensor标量，而不是Python标量。

        **参数：**

        - **index** (Union[None, int, tuple(int)]) - Tensor的索引。默认值：None。

        **返回：**

        Tensor标量，dtype与原始Tensor的相同。

        **异常：**

        - **ValueError** - `index` 的长度不等于Tensor的ndim。

    .. py:method:: itemset(*args)

        将标量插入到Tensor（并将标量转换为Tensor的数据类型）。

        至少有1个参数，并且最后一个参数被定义为设定值。
        Tensor.itemset(\*args)等同于 :math:`Tensor[args] = item` 。

        **参数：**

        **args** (Union[(numbers.Number), (int/tuple(int), numbers.Number)]) - 指定索引和值的参数。如果 `args` 包含一个参数（标量），则其仅在Tensor大小为1的情况下使用。如果 `args` 包含两个参数，则最后一个参数是要设置的值且必须是标量，而第一个参数指定单个Tensor元素的位置。参数值是整数或者元组。

        **返回：**

        一个新的Tensor，其值为 :math:`Tensor[args] = item` 。

        **异常：**

        - **ValueError** - 第一个参数的长度不等于Tensor的ndim。
        - **IndexError** - 只提供了一个参数，并且原来的Tensor不是标量。

    .. py:method:: itemsize
        :property:

        返回一个Tensor元素的长度（以字节为单位）。

    .. py:method:: masked_fill(mask, value)

        根据mask矩阵，将值为True对应位置填充为value值。mask矩阵的shape必须与原Tensor相同。

        **参数：**
            - **mask** (Tensor[bool]) - mask矩阵，值为bool类型的Tensor。
            - **value** (Union[int, float]) - 填充值，可以为int或float类型。

        **返回：**
            Tensor, shape和dtype与原Tensor相同。

        **异常：**
            - **TypeError** - mask不是Tensor。
            - **TypeError** - mask不是bool类型的Tensor。
            - **TypeError** - value不是int或float类型。

    .. py:method:: max(axis=None, keepdims=False, initial=None, where=True)

        返回Tensor的最大值或轴方向上的最大值。

        **参数：**

        - **axis** (Union[None, int, tuple of ints], optional) - 轴，在该轴方向上进行操作。默认情况下，使用扁平输入。如果该参数为整数元组，则在多个轴上选择最大值，而不是在单个轴或所有轴上进行选择。默认值：None。
        - **keepdims** (bool, optional) - 如果这个参数为True，被删去的维度保留在结果中，且维度大小设为1。有了这个选项，结果就可以与输入数组进行正确的广播运算。默认值：False。
        - **initial** (scalar, optional) - 输出元素的最小值。如果对空切片进行计算，则该参数必须设置。默认值：None。
        - **where** (bool Tensor, optional) - 一个bool数组，被广播以匹配数组维度和选择包含在降维中的元素。如果传递了一个非默认值，则还必须提供初始值。默认值：True。

        **返回：**

        Tensor或标量，输入Tensor的最大值。如果 `axis` 为None，则结果是一个标量值。如果提供了 `axis` ，则结果是Tensor ndim - 1维度的一个数组。

        **异常：**

        - **TypeError** - 参数具有前面未指定的类型。

    .. py:method:: mean(axis=(), keep_dims=False)

        返回指定维度上所有元素的均值，并降维。

        **参数：**

        - **axis** (Union[None, int, tuple(int), list(int)]) - 计算mean的维度。当 `axis` 为None或空元组时，计算所有维度。当 `axis` 为int、tuple(int)或list(int)时，记Tensor的维度为dim，则其取值范围为[-dim, dim)。默认值：()。
        - **keep_dims** (bool) - 计算结果是否保留维度。默认值：False。

        **返回：**

        与输入的张量具有相同的数据类型的Tensor。

    .. py:method:: min(axis=None, keepdims=False, initial=None, where=True)

        返回Tensor的最小值或轴方向上的最小值。

        **参数：**

        - **axis** (Union[None, int, tuple of ints], optional) - 轴，在该轴方向上进行操作。默认情况下，使用扁平输入。如果该参数为整数元组，则在多个轴上选择最小值，而不是在单个轴或所有轴上进行选择。默认值：None。
        - **keepdims** (bool, optional) - 如果这个参数为True，被删去的维度保留在结果中，且维度大小设为1。有了这个选项，结果就可以与输入数组进行正确的广播运算。默认值：False。
        - **initial** (scalar, optional) - 输出元素的最大值。如果对空切片进行计算，则该参数必须设置。默认值：None。
        - **where** (bool Tensor, optional) - 一个布尔数组，被广播以匹配数组维度和选择包含在降维中的元素。如果传递了一个非默认值，则还必须提供初始值。默认值：True。

        **返回：**

        Tensor或标量，输入Tensor的最小值。如果轴为None，则结果为一个标量值。如果提供了 `axis` ，则结果是Tensor.ndim - 1维度的一个数组。

        **异常：**

        - **TypeError** - 参数具有前面未指定的类型。

    .. py:method:: narrow(axis, start, length):

        沿指定轴，指定起始位置获取指定长度的Tensor。

        **参数：**
            - **axis** (int) - 指定的轴。
            - **start** (int) - 指定的起始位置。
            - **length** (int) - 指定的长度。

        **返回：**
            Tensor。

        **异常：**
            - **TypeError** - axis不是int类型。
            - **TypeError** - start不是int类型。
            - **TypeError** - length不是int类型。
            - **ValueError** - axis超出Tensor的维度范围。
            - **ValueError** - start超出Tensor的维度范围。
            - **ValueError** - start+length超出Tensor的维度范围。

    .. py:method:: nbytes
        :property:

        返回Tensor占用的总字节数。

    .. py:method:: ndim
        :property:

        返回Tensor维度的数量。

    .. py:method:: ptp(axis=None, keepdims=False)

        该函数名称是"peak to peak"的缩写。计算沿着axis的最大值与最小值的差值。

        .. note::
            不支持NumPy参数 `dtype` 和 `out` 。

        **参数：**

        - **axis** (Union[None, int, tuple(int)]) - 轴，在轴方向上可以计算范围。默认计算扁平数组的方差。默认值：None。
        - **keepdims** (bool) - 如果设为True，被删去的维度保留在结果中，且维度大小设为1。有了这个选项，结果将针对输入数组正确传递。默认值为False。

        **返回：**

        Tensor。

        **异常：**

        - **TypeError** - `self` 不是Tensor，或者 `axis` 和 `keepdims` 具有前面未指定的类型。

    .. py:method:: ravel()

        返回一个展开的一维Tensor。

        **返回：**

        一维Tensor，含有与输入相同的元素。

    .. py:method:: repeat(repeats, axis=None)

        对数组中的元素进行重复复制。

        **参数：**

        - **repeats** (Union[int, tuple, list]) - 每个元素的重复次数，`repeats` 被广播以适应指定轴的shape。
        - **axis** (int, optional) - 轴方向上的重复值。默认情况下，使用展开的输入Tensor，并返回一个展开的输出Tensor。

        **返回：**

        Tensor，除了维度外，与输入Tensor具有相同的shape。

        **异常：**

        - **ValueError** - 维度超出范围。
        - **TypeError** - 参数类型不匹配。

    .. py:method:: reshape(*shape)

        不改变数据的情况下，将Tensor的shape改为输入的新shape。

        **参数：**

        **shape** (Union[int, tuple(int), list(int)]) - 新的shape应与原来的shape兼容。如果参数值为整数，则结果是该长度的一维数组。shape的维度可以为-1。在这种情况下，将根据数组的长度和剩下的维度计算出该值。

        **返回：**

        Tensor，具有新shape的Tensor。

        **异常：**

        - **TypeError** - 新shape不是整数、列表或元组。
        - **ValueError** - 新shape与原来Tensor的shape不兼容。

    .. py:method:: resize(*new_shape)

        将Tensor改为输入的新shape, 并将不足的元素补0。

        .. note::
            此方法不更改输入数组的大小，也不返回NumPy中的任何内容，而是返回一个具有输入大小的新Tensor。不支持Numpy参数 `refcheck` 。

        **参数：**

        **new_shape** (Union[ints, tuple of ints]) - 指定Tensor的新shape。

        **返回：**

        Tensor。

    .. py:method:: searchsorted(v, side='left', sorter=None)

        查找应插入元素以保存顺序的位置索引。

        **参数：**

        - **v** (Union[int, float, bool, list, tuple, Tensor]) - 要插入元素的值。
        - **side** ('left', 'right', optional) - 如果参数值为'left'，则给出找到的第一个合适位置的索引。如果参数值为'right'，则返回最后一个这样的索引。如果没有合适的索引，则返回0或N（其中N是Tensor的长度）。默认值：'left'。
        - **sorter** (Union[int, float, bool, list, tuple, Tensor]) - 整数索引的可选一维数组，将Tensor按升序排序。它们通常是NumPy argsort方法的结果。

        **返回：**

        Tensor，shape与 `v` 相同的插入点数组。

        **异常：**

        - **ValueError** - `side` 或 `sorter` 的参数无效。

    .. py:method:: shape
        :property:

        返回Tensor的shape。

    .. py:method:: size
        :property:

        返回Tensor中的元素总数。

    .. py:method:: squeeze(axis=None)

        从Tensor中删除shape为1的维度。

        **参数：**

        **axis** (Union[None, int, list(int), tuple(int)], optional) - 选择shape中长度为1的条目的子集。如果选择shape条目长度大于1的轴，则报错。默认值为None。

        **返回：**

        Tensor，删除了长度为1的维度的全部子集或一个子集。

        **异常：**

        - **TypeError** - 输入的参数类型有误。
        - **ValueError** - 指定维度的shape大于1。

    .. py:method:: std(axis=None, ddof=0, keepdims=False)

        计算指定维度的标准差。
        标准差是方差的算术平方根，如：:math:`std = sqrt(mean(abs(x - x.mean())**2))` 。

        返回标准差。默认情况下计算展开数组的标准差，否则在指定维度上计算。

        .. note::
            不支持NumPy参数 `dtype` 、 `out` 和 `where` 。

        **参数：**

        - **axis** (Union[None, int, tuple(int)]) - 在该维度上计算标准差。默认值：`None` 。如果为 `None` ，则计算展开数组的标准偏差。
        - **ddof** (int) - δ自由度。计算中使用的除数是 :math:`N - ddof` ，其中 :math:`N` 表示元素的数量。默认值：0。
        - **keepdims** - 默认值：`False`。

        **返回：**

        含有标准差数值的Tensor。

    .. py:method:: strides
        :property:

        Tensor上每个维度跨度的字节元组。

    .. py:method:: sum(axis=None, dtype=None, keepdims=False, initial=None)

        返回指定维度上数组元素的总和。

        .. note::
            不支持NumPy参数 `out` 、 `where` 、 `casting` 、 `order` 、 `subok` 、 `signature` 和 `extobj` 。

        **参数：**

        - **axis** (Union[None, int, tuple(int)]) - 指定维度，在该维度方向上进行求和运算。默认值：None。如果参数值为None，会计算输入数组中所有元素的和。如果axis为负数，则从最后一维开始往第一维计算。如果axis为整数元组，会对该元组指定的所有轴方向上的元素进行求和。
        - **dtype** (`mindspore.dtype`, optional) - 默认值为None。会覆盖输出Tensor的dtype。
        - **keepdims** (bool) - 如果这个参数为True，被删去的维度保留在结果中，且维度大小设为1。有了这个选项，结果就可以与输入数组进行正确的广播运算。如果设为默认值，那么 `keepdims` 不会被传递给ndarray子类的sum方法。但是任何非默认值都会被传递。如果子类的方法未实现 `keepdims` ，则引发异常。默认值：False。
        - **initial** (scalar) - 初始化的起始值。默认值：None。

        **返回：**

        Tensor。具有与输入相同shape的Tensor，删除了指定的轴。如果输入Tensor是0维数组，或axis为None时，返回一个标量。

        **异常：**

        - **TypeError** - input不是Tensor，`axis` 不是整数或整数元组，`keepdims` 不是整数，或者 `initial` 不是标量。
        - **ValueError** - 任意轴超出范围或存在重复的轴。

    .. py:method:: swapaxes(axis1, axis2)

        交换Tensor的两个维度。

        **参数：**

        - **axis1** (int) - 第一个维度。
        - **axis2** (int) - 第二个维度。

        **返回：**

        转化后的Tensor，与输入具有相同的数据类型。

        **异常：**

        - **TypeError** - `axis1` 或 `axis2` 不是整数。
        - **ValueError** - `axis1` 或 `axis2` 不在 `[-ndim, ndim-1]` 范围内。

    .. py:method:: take(indices, axis=None, mode='clip')

        在指定维度上获取Tensor中的元素。

        **参数：**

        - **indices** (Tensor) - 待提取的值的shape为 `(Nj...)` 的索引。
        - **axis** (int, optional) - 在指定维度上选择值。默认情况下，使用展开的输入数组。默认值：None。
        - **mode** ('raise', 'wrap', 'clip', optional)

          - raise：抛出错误。
          - wrap：绕接。
          - clip：裁剪到范围。 `clip` 模式意味着所有过大的索引都会被在指定轴方向上指向最后一个元素的索引替换。注：这将禁用具有负数的索引。默认值：`clip` 。

        **返回：**

        Tensor，索引的结果。

        **异常：**

        - **ValueError** - `axis` 超出范围，或 `mode` 被设置为'raise'、'wrap'和'clip'以外的值。

    .. py:method:: to_tensor(slice_index=None, shape=None, opt_shard_group=None)

        返回init_data()的结果，并获取此Tensor的数据。

        .. note::
            不建议使用 `to_tensor`。请使用 `init_data` 。

        **参数：**

        - **slice_index** (int) - 参数切片的索引。在初始化参数切片的时候使用，保证使用相同切片的设备可以生成相同的Tensor。默认值：None。
        - **shape** (list[int]) - 切片的shape，在初始化参数切片时使用。默认值：None。
        - **opt_shard_group** (str) - 优化器分片组，在自动或半自动并行模式下用于获取参数切片的分片。默认值：None。

        **返回：**

        初始化的Tensor。

    .. py:method:: trace(offset=0, axis1=0, axis2=1, dtype=None)

        在Tensor的对角线方向上的总和。

        **参数：**

        - **offset** (int, optional) - 对角线与主对角线的偏移。可以是正值或负值。默认为主对角线。
        - **axis1** (int, optional) - 二维子数组的第一轴，对角线应该从这里开始。默认为第一轴(0)。
        - **axis2** (int, optional) - 二维子数组的第二轴，对角线应该从这里开始。默认为第二轴。
        - **dtype** (`mindspore.dtype` , optional) - 默认值为None。覆盖输出Tensor的dtype。

        **返回：**

        Tensor，对角线方向上的总和。

        **异常：**

        **ValueError** - 输入Tensor的维度少于2。

    .. py:method:: transpose(*axes)

        返回被转置后的Tensor。

        - 对于一维Tensor，这没有影响，因为转置后的向量是相同的。
        - 对于二维Tensor，是标准的矩阵转置。
        - 对于n维Tensor，如果提供了维度，则它们的顺序代表维度的置换方式。

        如果未提供轴，且Tensor.shape等于(i[0], i[1],...i[n-2], i[n-1])，则Tensor.transpose().shape等于(i[n-1], i[n-2], ... i[1], i[0])。

        **参数：**

        - **axes** (Union[None, tuple(int), list(int), int], optional) - 如果 `axes` 为None或未设置，则该方法将反转维度。如果 `axes` 为tuple(int)或list(int)，则Tensor.transpose()把Tensor转置为新的维度。如果 `axes` 为整数，则此表单仅作为元组/列表表单的备选。

        **返回：**

        Tensor，具有与输入Tensor相同的维度，其中维度被准确的排列。

        **异常：**

        - **TypeError** - 输入参数类型有误。
        - **ValueError** - `axes` 的数量不等于Tensor.ndim。

    .. py:method:: var(axis=None, ddof=0, keepdims=False)

        在指定维度上的方差。

        方差是平均值的平方偏差的平均值，即：:math:`var = mean(abs(x - x.mean())**2)` 。

        返回方差值。默认情况下计算展开Tensor的方差，否则在指定维度上计算。

        .. note::
            不支持NumPy参数 `dtype` 、 `out` 和 `where` 。

        **参数：**

        - **axis** (Union[None, int, tuple(int)]) - 维度，在指定维度上计算方差。其默认值是展开Tensor的方差。默认值：None。
        - **ddof** (int) - δ自由度。默认值：0。计算中使用的除数是 :math:`N - ddof` ，其中 :math:`N` 表示元素的数量。
        - **keepdims** (bool) - 默认值：False。

        **返回：**

        含有方差值的Tensor。

    .. py:method:: view(*shape)

        根据输入shape重新创建一个Tensor，与原Tensor数据相同。

        **参数：**

        **shape** (Union[tuple(int), int]) - 输出Tensor的维度。

        **返回：**

        Tensor，具有与输入shape相同的维度。
