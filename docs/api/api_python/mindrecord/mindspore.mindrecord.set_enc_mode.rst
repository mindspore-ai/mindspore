
.. py:function:: mindspore.mindrecord.set_enc_mode(enc_mode="AES-GCM")

    设置MindRecord数据格式的加密算法。

    参数：
        - **enc_mode** (Union[str, function], 可选) - 指定加密模式，当设置 `enc_key` 时启用。
          支持的加密选项有： ``"AES-GCM"`` 、 ``"AES-CBC"`` 、 ``"SM4-CBC"`` 和用户自定义加密算法。默认值： ``"AES-GCM"`` 。
          如果是自定义加密，用户需要自己保证加密正确性，并且在出错时报异常。

    异常：
        - **ValueError** - 参数 `enc_mode` 无效或者自定义加密不能被调用执行。
