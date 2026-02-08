mindspore.parallel.load_distributed_checkpoint
==============================================

.. py:function:: mindspore.parallel.load_distributed_checkpoint(network, checkpoint_filenames=None, predict_strategy=None, train_strategy_filename=None, strict_load=False, dec_key=None, dec_mode='AES-GCM', format='ckpt', unified_safetensors_dir=None, dst_safetensors_dir=None, rank_id=None, output_format='safetensors', name_map=None, max_process_num=64, return_param_dict=False)

    加载分布式checkpoint参数到网络，用于分布式推理。

    .. note::
        只有 `format` 设置为 `safetensors` 并且 `network` 为 `None` 时，output_format才会生效。

    参数：
        - **network** (Cell) - 分布式预测网络，format为 `safetensors` 时，network入参可以不传递或传递为None，此时接口执行保存模式。
        - **checkpoint_filenames** (list[str]) - checkpoint文件的名称，按rank id顺序排列。默认值： ``None`` 。
        - **predict_strategy** (Union[dict, str]) - 预测时参数的切分策略或者策略文件。默认值： ``None`` 。
        - **train_strategy_filename** (str) - 训练策略proto文件名。默认值： ``None`` 。
        - **strict_load** (bool) - 表示是否严格加载参数到网络。如果值为 ``False`` ，则当checkpoint文件中参数名称的后缀与网络中的参数相同时，加载参数到网络。当类型不一致时，对相同类型的参数进行类型转换，如从float32到float16。默认值： ``False`` 。
        - **dec_key** (Union[None, bytes]) - 用于解密的字节类型key。如果value为 ``None`` ，则不需要解密。默认值： ``None`` 。
        - **dec_mode** (str) - 指定解密模式，目前支持 ``'AES-GCM'`` ， ``'AES-CBC'`` 和 ``'SM4-CBC'`` 。仅当 `dec_key` 不设为 ``None`` 时，该参数有效。默认值： ``'AES-GCM'`` 。
        - **format** (str) - 待加载进网络的输入权重格式。可以设置为 "ckpt" 或 "safetensors"。默认值： ``"ckpt"`` 。
        - **unified_safetensors_dir** (str) - 待加载进网络的输入权重文件目录。默认值： ``None`` 。
        - **dst_safetensors_dir** (str) - 保存模式场景下，权重的保存目录。
        - **rank_id** (int) - 卡的逻辑序号。非保存模式下，通过初始化网络全局自动获取；保存模式下，按传入序号保存文件，若未传入，则全量保存。
        - **output_format** (str, 可选) - 控制转换后输出的 checkpoint 格式。可以设置为 "ckpt" 或 "safetensors"。默认值： ``"safetensors"``。
        - **name_map** (dict) - 权重映射字典，切分完的权重加载到网络或保存之前，会按照映射字典修改权重名字。默认值： ``None``。
        - **max_process_num** (int) - 最大进程数。默认值： ``64``。
        - **return_param_dict** (bool) - 是否返回 `param_dict`。默认值： ``False`` 。

    异常：
        - **TypeError** - 输入类型不符合要求。
        - **ValueError** - 无法加载checkpoint文件到网络。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst
