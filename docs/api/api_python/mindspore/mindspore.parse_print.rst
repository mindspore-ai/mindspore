mindspore.parse_print
=======================================

.. py:class:: mindspore.parse_print(print_file_name)

    解析由 mindspore.ops.Print 生成的数据文件。

    **参数：**

    **print_file_name** (str) – 需要解析的文件名。

    **返回：**

    List，由Tensor组成的list。

    **异常：**

    **ValueError** – 指定的文件不存在或为空。
    **RuntimeError** - 解析文件失败。

    **样例：**

    >>> import numpy as np
    >>> import mindspore.ops as ops
    >>> from mindspore.nn as nn
    >>> from mindspore import Tensor, context
    >>> context.set_context(mode=context.GRAPH_MODE, print_file_path='log.data')
    >>> class PrintInputTensor(nn.Cell):
    ...         def __init__(self):
    ...             super().__init__()
    ...             self.print = ops.Print()
    ...
    ...         def construct(self, input_pra):
    ...             self.print('print:', input_pra)
    ...             return input_pra

    >>> x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.float32)
    >>> input_pra = Tensor(x)
    >>> net = PrintInputTensor()
    >>> net(input_pra)

    >>> import mindspore
    >>> data = mindspore.parse_print('./log.data')
    >>> print(data)
    ['print:', Tensor(shape=[2, 4], dtype=Float32, value=
    [[ 1.00000000e+00,  2.00000000e+00,  3.00000000e+00,  4.00000000e+00],
    [ 5.00000000e+00,  6.00000000e+00,  7.00000000e+00,  8.00000000e+00]])]
