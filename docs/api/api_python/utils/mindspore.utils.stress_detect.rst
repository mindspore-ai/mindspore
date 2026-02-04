
.. py:function:: mindspore.utils.stress_detect(detect_type="aic")

    此接口将在后续版本中废弃，请使用接口 :func:`mindspore.tools.stress_detect` 代替。

    参数：
        - **detect_type** (str，可选) - 进行压测的类型。有两种可选：``'aic'`` 和 ``'hccs'``，分别会对设备进行AiCore和HCCS链路压测。默认 ``'aic'``。

    返回：
        int。返回值代表错误类型，0表示正常；1表示精度检测用例执行失败；2表示硬件故障，建议更换设备。
