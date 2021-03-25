
# It is still under development

# Contents

- [Contents](#contents)
- [PanGu1 Description](#bert-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
- [Script and Sample Code](#script-and-sample-code)
- [ModelZoo Homepage](#modelzoo-homepage)

# [PanGu1 Description](#contents)

We release the code to explore the new front-edge of training large model with billions or even trillions of parameters.
By MindSpore's parallel feature, we adopt the efficient model parallel and data parallel technology such as operator level parallelism,
to minimize the communication cost and maximize computation efficiency.
The code is easy to scale to thousands of NPUs and trillion parameters with little modifications.

In the mean while, we run our parallel training upon a language model, named PanGu1, to demonstrate the large model can be trained easily
with our parallel setting. We summarized the training tricks as followings:

1. Data Parallel
2. Model Parallel
3. Optimizer Parallel

The above features can be found [here](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.1/auto_parallel.html).
More amazing features are still under developing.

# [Model Architecture](#contents)

The PanGu1 model stacks many layers of transformer decoder with a few modifications. Hybrid parallelism is applied to maximize the device utilization
and reduce the communication consumption. The code demonstrate the training procedure on 32 Ascend card by 4-way data parallelism on the top 8 model parallelism.
Both 4-way data parallelism and 8-way model parallelism can be modified in the configuration to fit the variable scale of the cluster.

# [Dataset](#contents)

- Open Source Dataset.

The above dataset is preprocessed with 1024 tokens for each example. The default column key in dataset.py is `input_ids`.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training as follows:

```bash

# run distributed training example

bash scripts/run_distribute_training.sh /path/dataset /path/hccl.json 8

```

We recommend to run the code on 32 Ascend cards.

For distributed training, an hccl configuration file with JSON format needs to be created in advance.
Please follow the instructions in the link below:
https:gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
├── README.md
├── scripts
│         └── run_distribute_train.sh
├── src
│         ├── dataset.py
│         ├── pangu1.py
│         ├── pangu1_wrapcell.py
│         └── utils.py
└── train.py
```

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
