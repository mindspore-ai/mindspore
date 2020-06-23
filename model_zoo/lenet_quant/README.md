# LeNet Quantization Aware Training

## Description

Training LeNet with MNIST dataset in MindSpore with quantization aware training.

This is the simple and basic tutorial for constructing a network in MindSpore with quantization aware.

In this tutorial, you will:

1. Train a MindSpore fusion model for MNIST from scratch using `nn.Conv2dBnAct` and `nn.DenseBnAct`.
2. Fine tune the fusion model by applying the quantization aware training auto network converter API `convert_quant_network`, after the network convergence then export a quantization aware model checkpoint file.
3. Use the quantization aware model to create an actually quantized model for the Ascend inference backend.
4. See the persistence of accuracy in inference backend and a 4x smaller model. To see the latency benefits on mobile, try out the Ascend inference backend examples.


## Train fusion model

### Install

Install MindSpore base on the ascend device and GPU device from [MindSpore](https://www.mindspore.cn/install/en).


```python
pip uninstall -y mindspore-ascend
pip uninstall -y mindspore-gpu
pip install mindspore-ascend.whl
```

Then you will get the following display


```bash
>>> Found existing installation: mindspore-ascend
>>> Uninstalling mindspore-ascend:
>>>  Successfully uninstalled mindspore-ascend.
```

### Prepare Dataset

Download the MNIST dataset, the directory structure is as follows:

```
└─MNIST_Data
    ├─test
    │      t10k-images.idx3-ubyte
    │      t10k-labels.idx1-ubyte
    └─train
           train-images.idx3-ubyte
           train-labels.idx1-ubyte
```

### Define fusion model

Define a MindSpore fusion model using `nn.Conv2dBnAct` and `nn.DenseBnAct`.

```Python
class LeNet5(nn.Cell):
    """
    Define Lenet fusion model
    """

    def __init__(self, num_class=10, channel=1):
        super(LeNet5, self).__init__()
        self.num_class = num_class

        # change `nn.Conv2d` to `nn.Conv2dBnAct`
        self.conv1 = nn.Conv2dBnAct(channel, 6, 5, activation='relu')
        self.conv2 = nn.Conv2dBnAct(6, 16, 5, activation='relu')
        # change `nn.Dense` to `nn.DenseBnAct`
        self.fc1 = nn.DenseBnAct(16 * 5 * 5, 120, activation='relu')
        self.fc2 = nn.DenseBnAct(120, 84, activation='relu')
        self.fc3 = nn.DenseBnAct(84, self.num_class)

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
```

Get the MNIST from scratch dataset.

```Python
ds_train = create_dataset(os.path.join(args.data_path, "train"), 
                          cfg.batch_size, cfg.epoch_size)
step_size = ds_train.get_dataset_size()
```

### Train model

Load the Lenet fusion network, training network using loss `nn.SoftmaxCrossEntropyWithLogits` with optimization `nn.Momentum`.

```Python
# Define the network
network = LeNet5Fusion(cfg.num_classes)
# Define the loss
net_loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction="mean")
# Define optimization
net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)

# Define model using loss and optimization.
time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
config_ck = CheckpointConfig(save_checkpoint_steps=cfg.epoch_size * step_size,
                             keep_checkpoint_max=cfg.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
```

Now we can start training.

```Python
model.train(cfg['epoch_size'], ds_train, 
            callbacks=[time_cb, ckpoint_cb, LossMonitor()],
            dataset_sink_mode=args.dataset_sink_mode)
```

After all the following we will get the loss value of each step as following:

```bash
>>> Epoch: [  1/ 10] step: [  1/ 900], loss: [2.3040/2.5234], time: [1.300234]
>>> ...
>>> Epoch: [ 9/ 10] step: [887/ 900], loss: [0.0113/0.0223], time: [1.300234]
>>> Epoch: [ 9/ 10] step: [888/ 900], loss: [0.0334/0.0223], time: [1.300234]
>>> Epoch: [ 9/ 10] step: [889/ 900], loss: [0.0233/0.0223], time: [1.300234]
```

Also, you can just run this command instead.

```python
python train.py --data_path MNIST_Data --device_target Ascend
```

### Evaluate fusion model

After training epoch stop. We can get the fusion model checkpoint file like `checkpoint_lenet.ckpt`. Meanwhile, we can evaluate this fusion model.

```python
python eval.py --data_path MNIST_Data --device_target Ascend --ckpt_path checkpoint_lenet.ckpt
```

The top1 accuracy would display on shell.

```bash
>>> Accuracy: 98.53.
```

## Train quantization aware model

### Define quantization aware model

You will apply quantization aware training to the whole model and the layers of "fake quant op" are insert into the whole model. All layers are now perpare by "fake quant op".

Note that the resulting model is quantization aware but not quantized (e.g. the weights are float32 instead of int8).

```python
# define funsion network
network = LeNet5Fusion(cfg.num_classes)

# load quantization aware network checkpoint
param_dict = load_checkpoint(args.ckpt_path)
load_param_into_net(network, param_dict)

# convert funsion netwrok to quantization aware network
network = quant.convert_quant_network(network)
```

### load checkpoint

After convert to quantization aware network, we can load the checkpoint file.

```python
config_ck = CheckpointConfig(save_checkpoint_steps=cfg.epoch_size * step_size,
                             keep_checkpoint_max=cfg.keep_checkpoint_max)
ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
```

### train quantization aware model

Also, you can just run this command instread.

```python
python train_quant.py --data_path MNIST_Data --device_target Ascend --ckpt_path checkpoint_lenet.ckpt
```

After all the following we will get the loss value of each step as following:

```bash
>>> Epoch: [  1/ 10] step: [  1/ 900], loss: [2.3040/2.5234], time: [1.300234]
>>> ...
>>> Epoch: [ 9/ 10] step: [887/ 900], loss: [0.0113/0.0223], time: [1.300234]
>>> Epoch: [ 9/ 10] step: [888/ 900], loss: [0.0334/0.0223], time: [1.300234]
>>> Epoch: [ 9/ 10] step: [889/ 900], loss: [0.0233/0.0223], time: [1.300234]
```

### Evaluate quantization aware model

Procedure of quantization aware model evaluation is different from normal. Because the checkpoint was create by quantization aware model, so we need to load fusion model checkpoint before convert fusion model to quantization aware model.

```python
# define funsion network
network = LeNet5Fusion(cfg.num_classes)

# load quantization aware network checkpoint
param_dict = load_checkpoint(args.ckpt_path)
load_param_into_net(network, param_dict)

# convert funsion netwrok to quantization aware network
network = quant.convert_quant_network(network)
```

Also, you can just run this command insread.

```python
python eval_quant.py --data_path MNIST_Data --device_target Ascend --ckpt_path checkpoint_lenet.ckpt
```

The top1 accuracy would display on shell.

```bash
>>> Accuracy: 98.54.
```

## Note

Here are some optional parameters:

```bash
--device_target {Ascend,GPU,CPU}
    device where the code will be implemented (default: Ascend)
--data_path DATA_PATH
    path where the dataset is saved
--dataset_sink_mode DATASET_SINK_MODE
    dataset_sink_mode is False or True
```

You can run ```python train.py -h``` or ```python eval.py -h``` to get more information.

We encourage you to try this new capability, which can be particularly important for deployment in resource-constrained environments.