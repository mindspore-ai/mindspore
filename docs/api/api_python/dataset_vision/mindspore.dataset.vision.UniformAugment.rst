mindspore.dataset.vision.UniformAugment
=======================================

.. py:class:: mindspore.dataset.vision.UniformAugment(transforms, num_ops=2)

    从指定序列中均匀采样一批数据处理操作，并按顺序随机执行，即采样出的操作也可能不被执行。
    
    序列中的所有数据处理操作要求具有相同的输入和输出类型。后一个操作能够处理前一个操作的输出数据。

    参数：
        - **transforms** (Sequence) - 数据处理操作序列。
        - **num_ops** (int，可选) - 均匀采样的数据处理操作数。默认值： ``2`` 。

    异常：
        - **TypeError** - 当 `transforms` 的类型不为数据处理操作序列。
        - **TypeError** - 当 `num_ops` 的类型不为int。
        - **ValueError** - 当 `num_ops` 不为正数。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
