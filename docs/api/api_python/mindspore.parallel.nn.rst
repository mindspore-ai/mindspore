mindspore.parallel.nn
======================

Transformer接口的导入方式由 `mindspore.parallel.nn` 修改为 `mindspore.nn.transformer` ，这些接口的使用方式不变。

原始的导入方式将保留1-2个版本。你可以通过以下样例查看差异：

::

    # r1.5
    from mindspore.parallel.nn import Transformer

    # Current
    from mindspore.nn.transformer import Transformer
