
.. py:function:: mindspore.utils.stress_detect()

    用于检测硬件精度是否有故障。常见使用场景为在每个step或者保存checkpoint的时候，用户调用该接口，查看硬件是否有故障会影响精度。

    返回：
        int，返回值代表错误类型，0表示正常；非0表示硬件故障。
    
    支持平台：
        ``Ascend``
    
    **样例**：
    
        >>> from mindspore.utils import stress_detect
        >>> ret = stress_detect()
        >>> print(ret)
        0
