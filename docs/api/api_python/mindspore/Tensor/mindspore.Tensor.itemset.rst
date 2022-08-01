mindspore.Tensor.itemset
========================

.. py:method:: mindspore.Tensor.itemset(*args)

    将标量插入到Tensor（并将标量转换为Tensor的数据类型）。

    至少有1个参数，并且最后一个参数被定义为设定值。
    Tensor.itemset(\*args)等同于 :math:`Tensor[args] = item` 。

    参数：
        - **args** (Union[(numbers.Number), (int/tuple(int), numbers.Number)]) - 指定索引和值的参数。如果 `args` 包含一个参数（标量），则其仅在Tensor大小为1的情况下使用。如果 `args` 包含两个参数，则最后一个参数是要设置的值且必须是标量，而第一个参数指定单个Tensor元素的位置。参数值是整数或者元组。

    返回：
        一个新的Tensor，其值为 :math:`Tensor[args] = item` 。

    异常：
        - **ValueError** - 第一个参数的长度不等于Tensor的ndim。
        - **IndexError** - 只提供了一个参数，并且原来的Tensor不是标量。