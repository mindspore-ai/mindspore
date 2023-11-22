
.. py:function:: mindspore.mindrecord.set_enc_key(enc_key)

    设置MindRecord数据格式的加密密钥。

    参数：
        - **enc_key** (str) - 用于加密的密钥，有效长度为16、24或者32。其中：``None`` 代表不启用加密。

    异常：
        - **ValueError** - 参数 `enc_key` 不是字符串或者长度错误。
