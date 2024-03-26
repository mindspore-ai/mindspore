mindspore.load_checkpoint_async
===============================

.. py:function:: load_checkpoint_async(ckpt_file_name, net=None, strict_load=False, filter_prefix=None, dec_key=None, dec_mode="AES-GCM", specify_prefix=None, choice_func=None)

   异步加载checkpoint文件。

   .. warning::
       这是一个实验性API，后续可能修改或删除。

   .. note::
       - `specify_prefix` 和 `filter_prefix` 的功能相互之间没有影响。
       - 如果发现没有参数被成功加载，将会报ValueError。
       - `specify_prefix` 和 `filter_prefix` 参数已被弃用，推荐使用 `choice_func` 代替。并且使用这两个参数中的任何一个都将覆盖 `choice_func` 。

   参数：
       - **ckpt_file_name** (str) - checkpoint的文件名称。
       - **net** (Cell，可选) - 加载checkpoint参数的网络。默认值： ``None`` 。
       - **strict_load** (bool，可选) - 是否将严格加载参数到网络中。如果是 ``False`` ，它将根据相同的后缀名将参数字典中的参数加载到网络中，并会在精度不匹配时，进行强制精度转换，比如将float32转换为float16。默认值： ``False`` 。
       - **filter_prefix** (Union[str, list[str], tuple[str]]， 可选) - 废弃（请参考参数 `choice_func`）。以 `filter_prefix` 开头的参数将不会被加载。默认值： ``None`` 。
       - **dec_key** (Union[None, bytes]，可选) - 用于解密的字节类型密钥，如果值为 ``None`` ，则不需要解密。默认值： ``None`` 。
       - **dec_mode** (str，可选) - 该参数仅当 `dec_key` 不为 ``None`` 时有效。指定解密模式，目前支持 ``"AES-GCM"`` ， ``"AES-CBC"`` 和 ``"SM4-CBC"`` 。默认值： ``"AES-GCM"`` 。
       - **specify_prefix** (Union[str, list[str], tuple[str]]，可选) - 废弃（请参考参数 `choice_func`）。以 `specify_prefix` 开头的参数将会被加载。默认值： ``None`` 。
       - **choice_func** (Union[None, function]，可选) - 函数的输入值为字符串类型的Parameter名称，并且返回值是一个布尔值。如果返回 ``True`` ，则匹配自定义条件的Parameter将被加载。 如果返回 ``False`` ，则匹配自定义条件的Parameter将被删除。默认值： ``None`` 。

   返回：
       自定义的内部类， 调用其`result`方法可以得到 :func:`load_checkpoint` 返回的结果。

   异常：
       - **ValueError** - checkpoint文件格式不正确。
       - **ValueError** - 没有一个参数被成功加载。
       - **TypeError** - `specify_prefix` 或者 `filter_prefix` 的数据类型不正确。
