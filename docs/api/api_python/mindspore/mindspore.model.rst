mindspore.Model
================

   .. py:method:: infer_predict_layout(*predict_data)

      在 `AUTO_PARALLEL` 或 `SEMI_AUTO_PARALLEL` 模式下为预测网络生成参数layout，数据可以是单个或多个张量。

      .. note:: 同一批次数据应放在一个张量中。

      **参数：**

      **predict_data** (`Tensor`) – 单个或多个张量的预测数据。

      **返回：** 

      Dict，用于加载分布式checkpoint的参数layout字典。它总是作为 `load_distributed_checkpoint()` 函数的一个入参。

      **异常：**

      **RuntimeError** – 如果不是图模式（GRAPH_MODE）。

      **样例：**

      >>> # 该例子需要在多设备上运行。请参考mindpore.cn上的教程 > 分布式训练。
      >>> import numpy as np
      >>> import mindspore as ms
      >>> from mindspore import Model, context, Tensor
      >>> from mindspore.context import ParallelMode
      >>> from mindspore.communication import init
      >>> 
      >>> context.set_context(mode=context.GRAPH_MODE)
      >>> init()
      >>> context.set_auto_parallel_context(full_batch=True, parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
      >>> input_data = Tensor(np.random.randint(0, 255, [1, 1, 32, 32]), ms.float32)
      >>> model = Model(Net())
      >>> model.infer_predict_layout(input_data)

   .. py:method:: infer_train_layout(train_dataset, dataset_sink_mode=True, sink_size=-1)

      在 `AUTO_PARALLEL` 或 `SEMI_AUTO_PARALLEL` 模式下为训练网络生成参数layout，当前只有数据下沉模式可支持使用。

      .. warning:: 这是一个实验性的原型，可能会被改变和/或删除。

      .. note:: 这是一个预编译函数。参数必须与model.train()函数相同。

      **参数：**

      - **train_dataset** (`Dataset`) – 一个训练数据集迭代器。如果没有损失函数（loss_fn），返回一个包含多个数据的元组（data1, data2, data3, ...）并传递给网络。否则，返回一个元组（data, label），数据和标签将被分别传递给网络和损失函数。
      - **dataset_sink_mode** (`bool`) – 决定是否以数据集下沉模式进行训练。默认值：True。配置项是PyNative模式或CPU时，训练模型流程使用的是数据不下沉（non-sink）模式。默认值：True。
      - **sink_size** (`int`) – 控制每次数据下沉的数据量，如果 `sink_size` =-1，则每一次epoch下沉完整数据集。如果 `sink_size` >0，则每一次epoch下沉数据量为 `sink_size` 的数据集。如果 `dataset_sink_mode` 为False，则设置 `sink_size` 为无效。默认值：-1。

      **返回：** 

      Dict，用于加载分布式checkpoint的参数layout字典。

      **样例：**

      >>> # 该例子需要在多设备上运行。请参考mindpore.cn上的教程 > 分布式训练。
      >>> import numpy as np
      >>> import mindspore as ms
      >>> from mindspore import Model, context, Tensor, nn, FixedLossScaleManager
      >>> from mindspore.context import ParallelMode
      >>> from mindspore.communication import init
      >>> 
      >>> context.set_context(mode=context.GRAPH_MODE)
      >>> init()
      >>> context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
      >>> 
      >>> # 如何构建数据集，请参考官方网站上关于【数据集】的章节。
      >>> dataset = create_custom_dataset()
      >>> net = Net()
      >>> loss = nn.SoftmaxCrossEntropyWithLogits()
      >>> loss_scale_manager = FixedLossScaleManager()
      >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
      >>> model = Model(net, loss_fn=loss, optimizer=optim, metrics=None, loss_scale_manager=loss_scale_manager)
      >>> layout_dict = model.infer_train_layout(dataset)
