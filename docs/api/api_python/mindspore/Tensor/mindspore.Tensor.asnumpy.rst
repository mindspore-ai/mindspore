mindspore.Tensor.asnumpy
========================

.. py:method:: mindspore.Tensor.asnumpy()

    将张量转换为NumPy数组。该方法会将Tensor本身转换为NumPy的ndarray。这个Tensor和函数返回的ndarray共享内存地址。对Tensor本身的修改会反映到相应的ndarray上。

    返回：
        NumPy的ndarray，该ndarray与Tensor共享内存地址。