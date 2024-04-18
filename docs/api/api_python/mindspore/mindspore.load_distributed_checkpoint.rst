mindspore.load_distributed_checkpoint
======================================

.. py:function:: mindspore.load_distributed_checkpoint(network, checkpoint_filenames, predict_strategy=None, train_strategy_filename=None, strict_load=False, dec_key=None, dec_mode='AES-GCM')

    给分布式预测加载checkpoint文件到网络。用于分布式推理。关于分布式推理的细节，请参考： `分布式模型加载 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/model_loading.html>`_ 。

    参数：
        - **network** (Cell) - 分布式预测网络。
        - **checkpoint_filenames** (list[str]) - checkpoint文件的名称，按rank id顺序排列。
        - **predict_strategy** (dict) - 预测时参数的切分策略。默认值： ``None`` 。
        - **train_strategy_filename** (str) - 训练策略proto文件名。默认值： ``None`` 。
        - **strict_load** (bool) - 表示是否严格加载参数到网络。如果值为 ``False`` ，则当checkpoint文件中参数名称的后缀与网络中的参数相同时，加载参数到网络。当类型不一致时，对相同类型的参数进行类型转换，如从float32到float16。默认值： ``False`` 。
        - **dec_key** (Union[None, bytes]) - 用于解密的字节类型key。如果value为 ``None`` ，则不需要解密。默认值： ``None`` 。
        - **dec_mode** (str) - 仅当dec_key不设为 ``None`` 时，该参数有效。指定了解密模式，目前支持 ``'AES-GCM'`` ， ``'AES-CBC'`` 和 ``'SM4-CBC'`` 。默认值： ``'AES-GCM'`` 。

    异常：
        - **TypeError** - 输入类型不符合要求。
        - **ValueError** - 无法加载checkpoint文件到网络。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst
