
.. py:function:: mindspore.mindrecord.set_hash_mode(hash_mode)

    设置MindRecord数据格式的完整性校验算法。

    参数：
        - **hash_mode** (Union[str, function]) - 指定hash算法。
          支持的hash算法有： ``None`` 、 ``"sha224"`` 、 ``"sha256"`` 、 ``"sha384"`` 、 ``"sha512"`` 、
          ``"sha3_224"`` 、 ``"sha3_256"`` 、 ``"sha3_384"`` 、 ``"sha3_512"`` 和用户自定义hash算法。
          其中：``None`` 代表不启用文件完整性校验。

    异常：
        - **ValueError** - 参数 `hash_mode` 无效或者自定义hash算法不能被调用执行。
