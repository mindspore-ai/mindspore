
.. py:function:: mindspore.mindrecord.set_enc_mode(enc_mode="AES-GCM")

    设置MindRecord数据格式的加密算法。

    参数：
        - **enc_mode** (Union[str, function], 可选) - 指定加密模式，当设置 `enc_key` 时启用。
          支持的加密选项有： ``"AES-GCM"`` 、 ``"AES-CBC"`` 、 ``"SM4-CBC"`` 和用户自定义加密算法。默认值： ``"AES-GCM"`` 。
          其中：``None`` 代表不设置加密算法。

    异常：
        - **ValueError** - 参数 `enc_mode` 无效或者自定义加密不能被调用执行。
