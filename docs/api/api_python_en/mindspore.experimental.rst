mindspore.experimental
=======================

The experimental modules.

Experimental Optimizer
-----------------------

.. msplatformautosummary::
    :toctree: experimental/optim
    :nosignatures:
    :template: classtemplate.rst

    mindspore.experimental.optim.Optimizer
    mindspore.experimental.optim.Adam
    mindspore.experimental.optim.AdamW
    mindspore.experimental.optim.SGD


LRScheduler Class
^^^^^^^^^^^^^^^^^^

The dynamic learning rates in this module are all subclasses of LRScheduler, this module should be used with optimizers
in mindspore.experimental.optim, pass the optimizer instance to a LRScheduler when used. During the training process, the
LRScheduler subclass dynamically changes the learning rate by calling the `step` method.

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

.. msplatformautosummary::
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