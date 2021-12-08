mindspore.load
=======================================

.. py:class:: mindspore.load(file_name, **kwargs)

    加载MindIR文件。

    返回的对象可以由 `GraphCell` 执行，更多细节参见类 :class:`mindspore.nn.GraphCell` 。

    **参数：**

    - **file_name** (str) – MindIR文件名。
    - **kwargs** (dict) – 配置项字典。
      - **dec_key** (bytes) - 用于解密的字节类型密钥。 有效长度为 16、24 或 32。
      - **dec_mode** - 指定解密模式，设置dec_key时生效。可选项：'AES-GCM' | 'AES-CBC'。 默认值：“AES-GCM”。

    **返回：**

    Object，一个可以由 `GraphCell` 构成的可执行的编译图。

    **异常：**

    - **ValueError** – MindIR 文件名不正确。
    - **RuntimeError** - 解析MindIR文件失败。

    **样例：**

    >>> import numpy as np
    >>> import mindspore.nn as nn
    >>> from mindspore import Tensor, export, load
    >>>
    >>> net = nn.Conv2d(1, 1, kernel_size=3, weight_init="ones")
    >>> input_tensor = Tensor(np.ones([1, 1, 3, 3]).astype(np.float32))
    >>> export(net, input_tensor, file_name="net", file_format="MINDIR")
    >>> graph = load("net.mindir")
    >>> net = nn.GraphCell(graph)
    >>> output = net(input_tensor)
    >>> print(output)
    [[[[4. 6. 4.]
       [6. 9. 6.]
       [4. 6. 4.]]]]
