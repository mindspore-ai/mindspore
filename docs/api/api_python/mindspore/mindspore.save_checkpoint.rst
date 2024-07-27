mindspore.save_checkpoint
=========================

.. py:function:: mindspore.save_checkpoint(save_obj, ckpt_file_name, integrated_save=True, async_save=False, append_dict=None, enc_key=None, enc_mode="AES-GCM", choice_func=None, crc_check=False, format="ckpt", **kwargs)

    将网络权重保存到checkpoint文件中。

    .. note::
        `enc_mode` 和 `crc_check` 参数互斥，不能同时配置。

    参数：
        - **save_obj** (Union[Cell, list, dict]) - 待保存的对象。数据类型可为 :class:`mindspore.nn.Cell` 、list或dict。若为list，可以是 `Cell.trainable_params()` 的返回值，或元素为dict的列表（如[{"name": param_name, "data": param_data},…]，`param_name` 的类型必须是str，`param_data` 的类型必须是Parameter或者Tensor）；若为dict，可以是 `mindspore.load_checkpoint()` 的返回值。
        - **ckpt_file_name** (str) - checkpoint文件名称。如果文件已存在，将会覆盖原有文件。
        - **integrated_save** (bool) - 在并行场景下是否合并保存拆分的Tensor。默认值： ``True`` 。
        - **async_save** (bool) - 是否异步执行保存checkpoint文件。默认值： ``False`` 。
        - **append_dict** (dict) - 需要保存的其他信息。dict的键必须为str类型，dict的值类型必须是int、float、bool、string、Parameter或Tensor类型。默认值： ``None`` 。
        - **enc_key** (Union[None, bytes]) - 用于加密的字节类型密钥。如果值为 ``None`` ，那么不需要加密。默认值： ``None`` 。
        - **enc_mode** (str) - 该参数在 `enc_key` 不为 ``None`` 时有效，指定加密模式，目前仅支持 ``"AES-GCM"`` ， ``"AES-CBC"`` 和 ``"SM4-CBC"`` 。默认值： ``"AES-GCM"`` 。
        - **choice_func** (function) - 一个用于自定义控制保存参数的函数。函数的输入值为字符串类型的Parameter名称，并且返回值是一个布尔值。如果返回 ``True`` ，则匹配自定义条件的Parameter将被保存。 如果返回 ``False`` ，则未匹配自定义条件的Parameter不会被保存。默认值： ``None`` 。
        - **crc_check** (bool) - 是否在保存checkpoint时进行crc32校验，并把计算结果写到文件中。默认值： ``False`` 。
        - **format** (str) - 输出文件的格式，可以是 "ckpt" 或 "safetensors"。默认值：``"ckpt"``。
        - **kwargs** (dict) - 配置选项字典。

    异常：
        - **TypeError** - 如果参数 `save_obj` 类型不为 :class:`mindspore.nn.Cell` 、list或者dict。
        - **TypeError** - 如果参数 `integrated_save` 或 `async_save` 不是bool类型。
        - **TypeError** - 如果参数 `ckpt_file_name` 不是字符串类型。

    教程样例：
        - `保存与加载 - 保存和加载模型权重 <https://mindspore.cn/tutorials/zh-CN/master/beginner/save_load.html#保存和加载模型权重>`_