mindspore.Tensor.move_to
========================

.. py:method:: mindspore.Tensor.move_to(to, blocking=True)

    同步或异步的方式将Tensor拷贝到目标设备上，默认同步方式。只支持PyNative模式。

    参数：
        - **to** (str) - 字符串类型，取值为 ``"Ascend"``、 ``"GPU"``、 ``"CPU"`` 其中之一。
        - **blocking** (bool，可选) - 同步或者异步拷贝方式，默认为 ``True`` ，即同步拷贝。

    返回：
        存储在目标设备上的新Tensor，与原Tensor有相同的shape和type。

    异常：
        - **ValueError** - 如果 `blocking` 不是bool类型。
        - **ValueError** - 如果 `to` 不是 ``"Ascend"``、 ``"GPU"``、 ``"CPU"`` 其中之一。
        - **ValueError** - 如果执行模式不是PyNative模式。
