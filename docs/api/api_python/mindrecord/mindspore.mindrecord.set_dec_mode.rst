
.. py:function:: mindspore.mindrecord.set_dec_mode(dec_mode="AES-GCM")

    设置MindRecord数据格式的解密算法。

    如果使用的是内置 `enc_mode` ，且没有指定 `dec_mode` ，则使用 `enc_mode` 指定的加密算法进行解密。
    如果使用自定义加密函数，则必须在读取时指定自定义解密函数。

    参数：
        - **dec_mode** (Union[str, function], 可选) - 指定解密模式，当设置 `enc_key` 时启用。
          支持的解密选项有： ``"AES-GCM"`` 、 ``"AES-CBC"`` 、 ``"SM4-CBC"`` 和用户自定义解密算法。默认值： ``"AES-GCM"`` 。
          其中：``None`` 代表不设置解密算法。如果是自定义解密，用户需要自己保证解密正确性，并且在出错时报异常。

    异常：
        - **ValueError** - 参数 `dec_mode` 无效或者自定义解密不能被调用执行。
