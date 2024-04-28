mindspore.load_segmented_checkpoints
====================================

.. py:function:: mindspore.load_segmented_checkpoints(ckpt_file_dir, net=None, strict_load=False, filter_prefix=None, dec_key=None, dec_mode="AES-GCM", specify_prefix=None, choice_func=None)

    加载指定路径下所有checkpoint文件。当ckpt_file_dir指定路径下存在多个checkpoint文件时，会逐个读取各个checkpoint文件并返回合并后的字典。

    .. note::
        - `specify_prefix` 和 `filter_prefix` 参数已被弃用，推荐使用 `choice_func` 代替。`specify_prefix` 和 `filter_prefix` 的功能相互之间没有影响，并且使用这两个参数中的任何一个都将覆盖 `choice_func` 。
        - 如果发现没有参数被成功加载，将会报ValueError。

    参数：
        - **ckpt_file_dir** (str) - checkpoint的文件夹路径。
        - **net** (Cell) - 加载checkpoint参数的网络。默认值： ``None`` 。
        - **strict_load** (bool) - 加载参数到网络中时，是否根据参数名进行全词匹配以及数据类型自动转换。如果是 ``False`` ，它将根据相同的后缀名将参数字典中的参数加载到网络中，并会在精度不匹配时，进行强制精度转换，比如将 `float32` 转换为 `float16` 。默认值： ``False`` 。
        - **filter_prefix** (Union[str, list[str], tuple[str]]) - 废弃（请参考参数 `choice_func`）。以 `filter_prefix` 开头的参数将不会被加载。默认值： ``None`` 。
        - **dec_key** (Union[None, bytes]) - 用于解密的字节类型密钥，如果值为 ``None`` ，则不需要解密。默认值： ``None`` 。
        - **dec_mode** (str) - 该参数仅当 `dec_key` 不为 ``None`` 时有效。指定解密模式，目前支持 ``"AES-GCM"`` ， ``"AES-CBC"`` 和 ``"SM4-CBC"`` 。默认值： ``"AES-GCM"`` 。
        - **specify_prefix** (Union[str, list[str], tuple[str]]) - 废弃（请参考参数 `choice_func`）。以 `specify_prefix` 开头的参数将会被加载。默认值： ``None`` 。
        - **choice_func** (Union[None, function]) - 函数的输入值为字符串类型的Parameter名称，并且返回值是一个布尔值。如果返回 ``True`` ，则匹配自定义条件的Parameter将被加载。 如果返回 ``False`` ，则匹配自定义条件的Parameter将被删除。默认值： ``None`` 。

    返回：
        字典，key是参数名称，value是Parameter类型。当使用 :func:`mindspore.save_checkpoint` 的 `append_dict` 参数和 :class:`mindspore.train.CheckpointConfig` 的 `append_info` 参数保存
        checkpoint， `append_dict` 和 `append_info` 是dict类型，且它们的值value是string时，加载checkpoint得到的返回值是string类型，其它情况返回值均是Parameter类型。

    异常：
        - **TypeError** - ckpt_file_dir入参非字符串类型。
        - **ValueError** - checkpoint文件夹路径不存在或传入路径非文件夹。
        - **ValueError** - checkpoint文件格式不正确。
        - **ValueError** - 没有一个参数被成功加载。
        - **TypeError** - `specify_prefix` 或者 `filter_prefix` 的数据类型不正确。