mindspore.parallel.nn
======================

The import path of Transformer APIs have been modified from `mindspore.parallel.nn` to `mindspore.nn.transformer`, while the usage of these APIs stay unchanged. The original import path will retain one or two versions. You can view the changes using the examples described below

::

    # r1.5
    from mindspore.parallel.nn import Transformer

    # Current
    from mindspore.nn.transformer import Transformer
