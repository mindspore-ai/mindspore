mindspore.save_checkpoint
=========================

.. py:function:: mindspore.save_checkpoint(save_obj, ckpt_file_name, integrated_save=True, async_save=False, append_dict=None, enc_key=None, enc_mode="AES-GCM")

    将网络权重保存到checkpoint文件中。

    参数：
        - **save_obj** (Union[Cell, list]) - Cell对象或者数据列表（列表的每个元素为字典类型，比如[{"name": param_name, "data": param_data},…]，`param_name` 的类型必须是str，`param_data` 的类型必须是Parameter或者Tensor）。
        - **ckpt_file_name** (str) - checkpoint文件名称。如果文件已存在，将会覆盖原有文件。
        - **integrated_save** (bool) - 在并行场景下是否合并保存拆分的Tensor。默认值：True。
        - **async_save** (bool) - 是否异步执行保存checkpoint文件。默认值：False。
        - **append_dict** (dict) - 需要保存的其他信息。dict的键必须为str类型，dict的值类型必须是int、float、bool、string、Parameter或Tensor类型。默认值：None。
        - **enc_key** (Union[None, bytes]) - 用于加密的字节类型密钥。如果值为None，那么不需要加密。默认值：None。
        - **enc_mode** (str) - 该参数在 `enc_key` 不为None时有效，指定加密模式，目前仅支持"AES-GCM"和"AES-CBC"。默认值："AES-GCM"。

    异常：
        - **TypeError** - 如果参数 `save_obj` 类型不为nn.Cell或者list，且如果参数 `integrated_save` 及 `async_save` 非bool类型。如果参数 `ckpt_file_name` 不是字符串类型。
