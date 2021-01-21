# MindSpore Deep Probabilistic Programming

![MindSpore+ZhuSuan](https://images.gitee.com/uploads/images/2020/0814/172009_ff0cdc1a_6585083.png "MS-Zhusuan.PNG")

MindSpore Deep Probabilistic Programming (MDP) is a programming library for Bayesian deep learning. MDP is cooperatively developed with [ZhuSuan](https://zhusuan.readthedocs.io/en/latest/), which provides deep learning style primitives and algorithms for building probabilistic models and applying Bayesian inference.

The objective of MDP is to integrate deep learning with Bayesian learning. On the one hand, similar to other Deep Probabilistic Programming Languages (DPPL) (e.g., TFP, Pyro), for the professional Bayesian learning researchers, MDP provides probability sampling, inference algorithms, and model building libraries; On the other hand, MDP provides high-level APIs for DNN researchers that are unfamiliar with Bayesian models, making it possible to take advantage of Bayesian models without the need of changing their DNN programming logics.

## Layer 0: High performance kernels for different platforms

- Random sampling kernels;
- Mathematical kernels that are used by Bayesian models.

## Layer 1: Probabilistic Programming (PP) focuses on professional Bayesian learning

### Layer 1-1: Statistical distributions classes used to generate stochastic tensors

- Distributions ([mindspore.nn.probability.distribution](https://gitee.com/mindspore/mindspore/tree/master/mindspore/nn/probability/distribution)): A large collection of probability distributions.
- Bijectors([mindspore.nn.probability.bijectors](https://gitee.com/mindspore/mindspore/tree/master/mindspore/nn/probability/bijector)): Reversible and composable transformations of random variables.

### Layer 1-2: Probabilistic inference algorithms

- SVI([mindspore.nn.probability.infer.variational](https://gitee.com/mindspore/mindspore/tree/master/mindspore/nn/probability/infer/variational)): A unified interface for stochastic variational inference.
- MC: Algorithms for approximating integrals via sampling.

## Layer 2: Deep Probabilistic Programming (DPP) aims to provide composable BNN modules

- Layers([mindspore.nn.probability.bnn_layers](https://gitee.com/mindspore/mindspore/tree/master/mindspore/nn/probability/bnn_layers)): BNN layers, which are used to construct BNN.
- Dpn([mindspore.nn.probability.dpn](https://gitee.com/mindspore/mindspore/tree/master/mindspore/nn/probability/dpn)): A bunch of BNN models that allow to be integrated into DNN;
- Transform([mindspore.nn.probability.transforms](https://gitee.com/mindspore/mindspore/tree/master/mindspore/nn/probability/transforms)): Interfaces for the transformation between BNN and DNN;
- Context: context managers for models and layers.

## Layer 3: Toolbox provides a set of BNN tools for some specific applications

- Uncertainty Estimation([mindspore.nn.probability.toolbox.uncertainty_evaluation](https://gitee.com/mindspore/mindspore/tree/master/mindspore/nn/probability/toolbox/uncertainty_evaluation.py)): Interfaces to estimate epistemic uncertainty and aleatoric uncertainty.
- OoD detection: Interfaces to detect out of distribution samples.

![Structure of MDP](https://images.gitee.com/uploads/images/2020/0820/115117_2a20da64_7825995.png "MDP.png")
MDP requires MindSpore version 0.7.0-beta or later. MDP is actively evolving. Interfaces may change as Mindspore releases are iteratively updated.

## Tutorial

### Bayesian Neural Network

1. Process the required dataset. The MNIST dateset is used in the example. Data processing is consistent with [Implementing an Image Classification Application](https://www.mindspore.cn/tutorial/training/en/master/quick_start/quick_start.html) in Tutorial.

2. Define a Bayesian Neural Network. The bayesian LeNet is used in this example.

```python
import mindspore.nn as nn
from mindspore.nn.probability import bnn_layers
import mindspore.ops.operations as P

class BNNLeNet5(nn.Cell):
    """
    bayesian Lenet network

    Args:
        num_class (int): Num classes. Default: 10.

    Returns:
        Tensor, output tensor
    Examples:
        >>> BNNLeNet5(num_class=10)

    """
    def __init__(self, num_class=10):
        super(BNNLeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = bnn_layers.ConvReparam(1, 6, 5, stride=1, padding=0, has_bias=False, pad_mode="valid")
        self.conv2 = bnn_layers.ConvReparam(6, 16, 5, stride=1, padding=0, has_bias=False, pad_mode="valid")
        self.fc1 = bnn_layers.DenseReparam(16 * 5 * 5, 120)
        self.fc2 = bnn_layers.DenseReparam(120, 84)
        self.fc3 = bnn_layers.DenseReparam(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
```

The way to construct Bayesian Neural Network by bnn_layers is the same as DNN. It's worth noting that bnn_layers and traditional layers of DNN can be combined with each other.

3. Define the Loss Function and Optimizer

The loss function `SoftmaxCrossEntropyWithLogits` and the optimizer `AdamWeightDecay` are used in the example. Call the loss function and optimizer in the `__main__` function.

```python
if __name__ == "__main__":
    ...
    # define the loss function
    criterion = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = AdamWeightDecay(params=network.trainable_params(), learning_rate=0.0001)
    ...
```

4. Train the Network

The process of Bayesian network training is basically the same as that of DNN, the only difference is that WithLossCell is replaced with WithBNNLossCell suitable for BNN.
Based on the two parameters `backbone` and `loss_fn` in WithLossCell, WithBNNLossCell adds two parameters of `dnn_factor` and `bnn_factor`. Those two parameters are used to trade off backbone's loss and kl loss to prevent kl loss from being too large to cover backbone's loss.

```python
from mindspore.nn import TrainOneStepCell

if __name__ == "__main__":
    ...
    net_with_loss = bnn_layers.WithBNNLossCell(network, criterion, dnn_factor=60000, bnn_factor=0.000001)
    train_bnn_network = TrainOneStepCell(net_with_loss, optimizer)
    train_bnn_network.set_train()

    train_set = create_dataset('./mnist_data/train', 64, 1)
    test_set = create_dataset('./mnist_data/test', 64, 1)

    epoch = 100

    for i in range(epoch):
        train_loss, train_acc = train_model(train_bnn_network, network, train_set)

        valid_acc = validate_model(network, test_set)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tvalidation Accuracy: {:.4f}'.format(i, train_loss, train_acc, valid_acc))
```

The `train_model` and `validate_model` are defined as follows:

```python
import numpy as np

def train_model(train_net, net, dataset):
    accs = []
    loss_sum = 0
    for _, data in enumerate(dataset.create_dict_iterator()):
        train_x = Tensor(data['image'].astype(np.float32))
        label = Tensor(data['label'].astype(np.int32))
        loss = train_net(train_x, label)
        output = net(train_x)
        log_output = P.LogSoftmax(axis=1)(output)
        acc = np.mean(log_output.asnumpy().argmax(axis=1) == label.asnumpy())
        accs.append(acc)
        loss_sum += loss.asnumpy()

    loss_sum = loss_sum / len(accs)
    acc_mean = np.mean(accs)
    return loss_sum, acc_mean


def validate_model(net, dataset):
    accs = []
    for _, data in enumerate(dataset.create_dict_iterator()):
        train_x = Tensor(data['image'].astype(np.float32))
        label = Tensor(data['label'].astype(np.int32))
        output = net(train_x)
        log_output = P.LogSoftmax(axis=1)(output)
        acc = np.mean(log_output.asnumpy().argmax(axis=1) == label.asnumpy())
        accs.append(acc)

    acc_mean = np.mean(accs)
    return acc_mean
```

### Variational Inference

1. Define the Variational Auto-Encoder, we only need to self-define the encoder and decoder(DNN model).

```python
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.nn.probability.dpn import VAE

class Encoder(nn.Cell):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Dense(1024, 800)
        self.fc2 = nn.Dense(800, 400)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x


class Decoder(nn.Cell):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Dense(400, 1024)
        self.sigmoid = nn.Sigmoid()
        self.reshape = P.Reshape()

    def construct(self, z):
        z = self.fc1(z)
        z = self.reshape(z, IMAGE_SHAPE)
        z = self.sigmoid(z)
        return z


encoder = Encoder()
decoder = Decoder()
vae = VAE(encoder, decoder, hidden_size=400, latent_size=20)
```

2. Use ELBO interface to define the loss function and define the optimizer, then construct the cell_net using WithLossCell.

```python
from mindspore.nn.probability.infer import ELBO

net_loss = ELBO(latent_prior='Normal', output_prior='Normal')
optimizer = nn.Adam(params=vae.trainable_params(), learning_rate=0.001)
net_with_loss = nn.WithLossCell(vae, net_loss)
```

3. Process the required dataset. The MNIST dateset is used in the example. Data processing is consistent with [Implementing an Image Classification Application](https://www.mindspore.cn/tutorial/training/en/master/quick_start/quick_start.html) in Tutorial.
4. Use SVI interface to train VAE network. vi.run can return the trained network, get_train_loss can get the loss after training.

```python
from mindspore.nn.probability.infer import SVI

vi = SVI(net_with_loss=net_with_loss, optimizer=optimizer)
vae = vi.run(train_dataset=ds_train, epochs=10)
trained_loss = vi.get_train_loss()
```

5. Use the trained VAE network, we can generate new samples or reconstruct the input samples.

```python
IMAGE_SHAPE = (-1, 1, 32, 32)
generated_sample = vae.generate_sample(64, IMAGE_SHAPE)
for sample in ds_train.create_dict_iterator():
    sample_x = Tensor(sample['image'], dtype=mstype.float32)
    reconstructed_sample = vae.reconstruct_sample(sample_x)
```

### Transform DNN to BNN

For DNN researchers who are unfamiliar with Bayesian models, MDP provides high-level APIs `TransformToBNN` to support one-click conversion of DNN models to BNN models.

1. Define a Deep Neural Network. The LeNet is used in this example.

```python
from mindspore.common.initializer import TruncatedNormal
import mindspore.nn as nn
import mindspore.ops.operations as P

def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    """
    Lenet network

    Args:
        num_class (int): Num classes. Default: 10.

    Returns:
        Tensor, output tensor
    Examples:
        >>> LeNet5(num_class=10)

    """
    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.reshape = P.Reshape()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
```

2. Wrap DNN by TrainOneStepCell

```python
from mindspore.nn import WithLossCell, TrainOneStepCell

if __name__ == "__main__":
    network = LeNet5()

    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    optimizer = nn.AdamWeightDecay(params=network.trainable_params(), learning_rate=0.0001)

    net_with_loss = WithLossCell(network, criterion)
    train_network = TrainOneStepCell(net_with_loss, optimizer)
```

3. Instantiate class `TransformToBNN`

The `__init__` of `TransformToBNN` are as follows:

```python
class TransformToBNN:
    def __init__(self, trainable_dnn, dnn_factor=1, bnn_factor=1):
        net_with_loss = trainable_dnn.network
        self.optimizer = trainable_dnn.optimizer
        self.backbone = net_with_loss.backbone_network
        self.loss_fn = getattr(net_with_loss, "_loss_fn")
        self.dnn_factor = dnn_factor
        self.bnn_factor = bnn_factor
        self.bnn_loss_file = None
```

The arg `trainable_dnn` specifies a trainable DNN model wrapped by TrainOneStepCell, `dnn_factor` is the coefficient of backbone's loss, which is computed by loss function, and `bnn_factor` is the coefficient of kl loss, which is kl divergence of Bayesian layer. `dnn_factor` and `bnn_factor` are used to trade off backbone's loss and kl loss to prevent kl loss from being too large to cover backbone's loss.

```python
from mindspore.nn.probability import transforms

if __name__ == "__main__":
    ...
    bnn_transformer = transforms.TransformToBNN(train_network, 60000, 0.000001)
    ...
```

3-1. Transform the whole model
The method `transform_to_bnn_model` can transform both convolutional layer and full connection layer of DNN model to BNN model. Its code is as follows:

```python
    def transform_to_bnn_model(self,
                               get_dense_args=lambda dp: {"in_channels": dp.in_channels, "has_bias": dp.has_bias,
                                                          "out_channels": dp.out_channels, "activation": dp.activation},
                               get_conv_args=lambda dp: {"in_channels": dp.in_channels, "out_channels": dp.out_channels,
                                                         "pad_mode": dp.pad_mode, "kernel_size": dp.kernel_size,
                                                         "stride": dp.stride, "has_bias": dp.has_bias,
                                                         "padding": dp.padding, "dilation": dp.dilation,
                                                         "group": dp.group},
                               add_dense_args=None,
                               add_conv_args=None):
        r"""
        Transform the whole DNN model to BNN model, and wrap BNN model by TrainOneStepCell.

        Args:
            get_dense_args (function): The arguments gotten from the DNN full connection layer. Default: lambda dp:
                {"in_channels": dp.in_channels, "out_channels": dp.out_channels, "has_bias": dp.has_bias}.
            get_conv_args (function): The arguments gotten from the DNN convolutional layer. Default: lambda dp:
                {"in_channels": dp.in_channels, "out_channels": dp.out_channels, "pad_mode": dp.pad_mode,
                "kernel_size": dp.kernel_size, "stride": dp.stride, "has_bias": dp.has_bias}.
            add_dense_args (dict): The new arguments added to BNN full connection layer. Default: {}.
            add_conv_args (dict): The new arguments added to BNN convolutional layer. Default: {}.

        Returns:
            Cell, a trainable BNN model wrapped by TrainOneStepCell.
       """
```

Arg `get_dense_args` specifies which arguments to be gotten from full connection layer of DNN. Its Default value contains arguments common to nn.Dense and DenseReparameterization. Arg `get_conv_args` specifies which arguments to be gotten from convolutional layer of DNN. Its Default value contains arguments common to nn.Con2d and ConvReparameterization. Arg `add_dense_args` and `add_conv_args` specify which arguments to be add to full connection layer and convolutional layer of BNN. Note that the parameters in `add_dense_args` cannot be repeated with `get_dense_args`, so do `add_conv_args` and `get_conv_args`.

```python
if __name__ == "__main__":
    ...
    train_bnn_network = bnn_transformer.transform_to_bnn_model()
    ...
```

3-2. Transform a specific type of layers
The method `transform_to_bnn_layer` can transform a specific type of layers (nn.Dense or nn.Conv2d) in DNN model to corresponding BNN layer. Its code is as follows:

```python
    def transform_to_bnn_layer(self, dnn_layer, bnn_layer, get_args=None, add_args=None):
        r"""
        Transform a specific type of layers in DNN model to corresponding BNN layer.

        Args:
            dnn_layer_type (Cell): The type of DNN layer to be transformed to BNN layer. The optional values are
            nn.Dense, nn.Conv2d.
            bnn_layer_type (Cell): The type of BNN layer to be transformed to. The optional values are
                DenseReparameterization, ConvReparameterization.
            get_args (dict): The arguments gotten from the DNN layer. Default: None.
            add_args (dict): The new arguments added to BNN layer. Default: None.

        Returns:
            Cell, a trainable model wrapped by TrainOneStepCell, whose sprcific type of layer is transformed to the corresponding bayesian layer.
        """
```

Arg `dnn_layer` specifies which type of DNN layer to be transformed to BNN layer. The optional values are nn.Dense and nn.Conv2d. Arg `bnn_layer` specifies which type of BNN layer to be transformed to. The value should correspond to dnn_layer. Arg `get_args` and `add_args` specify the arguments gotten from DNN layer and the new arguments added to BNN layer respectively.

```python
if __name__ == "__main__":
    ...
    train_bnn_network = bnn_transformer.transform_to_bnn_layer()
    ...
```

### Uncertainty Evaluation

The uncertainty estimation toolbox is based on MindSpore Deep Probabilistic Programming (MDP), and it is suitable for mainstream deep learning models, such as regression, classification, target detection and so on. In the inference stage, with the uncertainy estimation toolbox, developers only need to pass in the trained model and training dataset, specify the task and the samples to be estimated, then can obtain the aleatoric uncertainty and epistemic uncertainty. Based the uncertainty information, developers can understand the model and the dataset better.

In classification task, for example, the model is lenet model. The MNIST dateset is used in the example. Data processing is consistent with [Implementing an Image Classification Application](https://www.mindspore.cn/tutorial/training/en/master/quick_start/quick_start.html) in Tutorial. For evaluating the uncertainty of test examples, the use of the toolbox is as follows:

```python
from mindspore.nn.probability.toolbox.uncertainty_evaluation import UncertaintyEvaluation
from mindspore.train.serialization import load_checkpoint, load_param_into_net

network = LeNet5()
param_dict = load_checkpoint('checkpoint_lenet.ckpt')
load_param_into_net(network, param_dict)
# get train and eval dataset
ds_train = create_dataset('workspace/mnist/train')
ds_eval = create_dataset('workspace/mnist/test')
evaluation = UncertaintyEvaluation(model=network,
                                   train_dataset=ds_train,
                                   task_type='classification',
                                   num_classes=10,
                                   epochs=1,
                                   epi_uncer_model_path=None,
                                   ale_uncer_model_path=None,
                                   save_model=False)
for eval_data in ds_eval.create_dict_iterator():
    eval_data = Tensor(eval_data['image'], mstype.float32)
    epistemic_uncertainty = evaluation.eval_epistemic_uncertainty(eval_data)
    aleatoric_uncertainty = evaluation.eval_aleatoric_uncertainty(eval_data)
```

## Examples

Examples in [mindspore/tests/st/probability](https://gitee.com/mindspore/mindspore/blob/master/tests/st/probability) are as follows:

- [Bayesian LeNet](https://gitee.com/mindspore/mindspore/blob/master/tests/st/probability/bnn_layers/test_bnn_layer.py). How to construct and train a LeNet by bnn layers.
- [Transform whole DNN model to BNN](https://gitee.com/mindspore/mindspore/blob/master/tests/st/probability/transforms/test_transform_bnn_model.py): How to transform whole DNN model to BNN.
- [Transform DNN layer to BNN](https://gitee.com/mindspore/mindspore/blob/master/tests/st/probability/transforms/test_transform_bnn_layer.py): How to transform one certainty type of layer in DNN model to corresponding Bayesian layer.
- [Variational Auto-Encoder](https://gitee.com/mindspore/mindspore/blob/master/tests/st/probability/dpn/test_gpu_svi_vae.py): Variational Auto-Encoder (VAE) model trained with MNIST to generate sample images.
- [Conditional Variational Auto-Encoder](https://gitee.com/mindspore/mindspore/blob/master/tests/st/probability/dpn/test_gpu_svi_cvae.py): Conditional Variational Auto-Encoder (CVAE) model trained with MNIST to generate sample images.
- [VAE-GAN](https://gitee.com/mindspore/mindspore/blob/master/tests/st/probability/dpn/test_gpu_vae_gan.py): VAE-GAN model trained with MNIST to generate sample images.
- [Uncertainty Estimation](https://gitee.com/mindspore/mindspore/blob/master/tests/st/probability/toolbox/test_uncertainty.py): Evaluate uncertainty of model and data.

## Community

As part of MindSpore, we are committed to creating an open and friendly environment.

- [Gitee](https://gitee.com/mindspore/mindspore/issues): Report bugs or make feature requests.
