
.. py:function:: mindspore.tools.stress_detect(detect_type="aic")

    用于检测硬件精度或链路间的通讯是否有故障。常见使用场景为在每个step或者保存checkpoint的时候，另起线程或通过Callback函数调用该接口，查看硬件是否有故障会影响精度。

    参数：
        - **detect_type** (str，可选) - 进行压测的类型。有两种可选：``'aic'`` 和 ``'hccs'``，分别会对设备进行AiCore和HCCS链路压测。默认 ``'aic'``。

    返回：
        int。返回值代表错误类型，0表示正常；1表示精度检测用例执行失败；2表示硬件故障，建议更换设备。
