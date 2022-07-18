mindspore.nn.probability.infer.SVI
==================================

.. py:class:: mindspore.nn.probability.infer.SVI(net_with_loss, optimizer)

    随机变分推断（Stochastic Variational Inference）。

    变分推断将推理问题转换为优化问题。隐藏变量上的一些分布由一组自由参数索引，这些参数经过优化，使分布成为最接近正确的后验分布。更多详细信息，请参阅
    `变分推理：统计学家评论 <https://arxiv.org/abs/1601.00670>`_。

    参数：
        - **net_with_loss** (Cell) - 具有损失函数的单元格。
        - **optimizer** (Cell) - 更新权重的优化器。

    .. py:method:: get_train_loss()

        返回：
            numpy.dtype，训练损失。

    .. py:method:: run(train_dataset, epochs=10)
     
        通过训练概率网络来优化参数，并返回训练好的网络。
    
        参数：
            - **train_dataset** (Dataset) - 训练数据集迭代器。
            - **epochs** (int) - 数据的迭代总数。默认值：10。
 
        返回：
            Cell，经过训练的概率网络。