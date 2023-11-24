mindspore.experimental
=======================

实验性模块。

实验性优化器
------------

.. mscnplatformautosummary::
    :toctree: experimental/optim
    :nosignatures:
    :template: classtemplate.rst

    mindspore.experimental.optim.Optimizer
    mindspore.experimental.optim.Adam
    mindspore.experimental.optim.AdamW
    mindspore.experimental.optim.SGD

LRScheduler类
^^^^^^^^^^^^^^^^

本模块中的动态学习率都是LRScheduler的子类，此模块仅与mindspore.experimental.optim下的优化器配合使用，使用时将优化器实例传递给LRScheduler类。在训练过程中，LRScheduler子类通过调用 `step` 方法进行学习率的动态改变。

.. code-block::

    import mindspore
    from mindspore import nn
    from mindspore.experimental import optim
    # Define the network structure of LeNet5. Refer to
    # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py

    net = LeNet5()
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    optimizer = optim.Adam(net.trainable_params(), lr=0.05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    def forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss, logits
    grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        optimizer(grads)
        return loss
    for epoch in range(6):
        # Create the dataset taking MNIST as an example. Refer to
        # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/mnist.py

        for data, label in create_dataset(need_download=False):
            train_step(data, label)
        scheduler.step()

.. mscnplatformautosummary::
    :toctree: experimental/optim
    :nosignatures:
    :template: classtemplate.rst

    mindspore.experimental.optim.lr_scheduler.LRScheduler
    mindspore.experimental.optim.lr_scheduler.ConstantLR
    mindspore.experimental.optim.lr_scheduler.CosineAnnealingLR
    mindspore.experimental.optim.lr_scheduler.CosineAnnealingWarmRestarts
    mindspore.experimental.optim.lr_scheduler.CyclicLR
    mindspore.experimental.optim.lr_scheduler.ExponentialLR
    mindspore.experimental.optim.lr_scheduler.LambdaLR
    mindspore.experimental.optim.lr_scheduler.LinearLR
    mindspore.experimental.optim.lr_scheduler.MultiplicativeLR
    mindspore.experimental.optim.lr_scheduler.MultiStepLR
    mindspore.experimental.optim.lr_scheduler.PolynomialLR
    mindspore.experimental.optim.lr_scheduler.ReduceLROnPlateau
    mindspore.experimental.optim.lr_scheduler.SequentialLR
    mindspore.experimental.optim.lr_scheduler.StepLR