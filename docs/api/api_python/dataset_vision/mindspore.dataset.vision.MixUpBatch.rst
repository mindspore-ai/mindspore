mindspore.dataset.vision.MixUpBatch
===================================

.. py:class:: mindspore.dataset.vision.MixUpBatch(alpha=1.0)

    对输入批次的图像和标注应用混合转换。从批处理中随机抽取两个图像，其中一个图像乘以随机权重 (lambda)，另一个图像乘以 (1 - lambda)，并相加。该处理将会同时应用于one-hot标注。

    上述的 lambda 是根据指定的参数 `alpha` 生成的。计算方式为在 [alpha, 1] 范围内随机生成两个系数 x1，x2 ，然后 lambda = (x1 / (x1 + x2))。

    请注意，在调用此处理之前，您需要将标注制作成 one-hot 格式并进行batch操作。

    参数：
        - **alpha** (float, 可选) - β分布的超参数，该值必须为正。默认值： ``1.0`` 。

    异常：
        - **TypeError** - 如果 `alpha` 不是float类型。
        - **ValueError** - 如果 `alpha` 不是正数。
        - **RuntimeError** - 如果输入图像的shape不是 <N, H, W, C> 或 <N, C, H, W>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
