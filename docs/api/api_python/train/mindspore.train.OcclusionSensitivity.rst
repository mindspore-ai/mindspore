mindspore.train.OcclusionSensitivity
=============================================

.. py:class:: mindspore.train.OcclusionSensitivity(pad_val=0.0, margin=2, n_batch=128, b_box=None)

    用于计算神经网络对给定图像的遮挡灵敏度（Occlusion Sensitivity），表示了图像的哪些部分对神经网络的分类决策最重要。

    遮挡敏感度是指神经网络对图像的类别预测概率如何随着图像被遮挡部分的变化而变化。遮挡敏感度值越高，意味着模型对类别预测的概率值下降越大，说明遮挡区域在神经网络的分类决策过程中越重要。

    参数：
        - **pad_val** (float) - 图像中被遮挡部分的填充值。默认值：0.0。
        - **margin** (Union[int, Sequence]) - 在要遮挡的像素点周围设置的长方体/立方体。默认值：2。
        - **n_batch** (int) - 一个batch中样本的数量。默认值：128。
        - **b_box** (Sequence) - 执行分析的目标区域的边界框(Bounding box)，其大小与输出图像的大小相匹配。如果没有设置此入参，Bounding box将与输入图像的大小相同；如果设置了此入参，输入图像将被裁剪为此大小，此设置值应形如：``[min1, max1, min2, max2,...]``，分别对应除batch size外各维度的最大最小值。默认值：None。

    .. py:method:: clear()

        内部评估结果清零。

    .. py:method:: eval()

         计算遮挡敏感度。

         返回：
             numpy ndarray。计算得到的遮挡敏感度值。

         异常：
             - **RuntimeError** - 如果没有先调用update方法，则会报错。

    .. py:method:: update(*inputs)

        更新inputs，包括 `model` 、 `y_pred` 和 `label` 。

        参数：
            - **inputs** - `y_pred` 和 `label` 为Tensor，list或numpy.ndarray，`y_pred` 是要测试的图像，一般为2D或3D，`label` 是用于检测神经网络预测值变化的类别标签，通常情况下为真实标签。`model` 为神经网络模型。

        异常：
            - **ValueError** - 输入数量不是3。
            - **RuntimeError** - `y_pred.shape[0]` 不是1。
            - **RuntimeError** - 标签数量与batch数量不同。
