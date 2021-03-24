
# It is still under development

# Contents

- [Contents](#contents)
- [GPT Description](#bert-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
- [Script and Sample Code](#script-and-sample-code)
- [ModelZoo Homepage](#modelzoo-homepage)

# [GPT Description](#contents)

The GPT network was proposed by OpenAI and it has three versions, i.e., GPT, GPT2 and GPT3. The newest version GPT3 was proposed in Jul 2020 and it is quite a large language model with 175 billion parameters. Stacking many Decoder structure of Transformer and feeding massive amount of training data, GPT3 becomes such a powerful language model that no fine-tuning process is needed. As the papre title says, language models are few-shot learners, GPT3 proves that with a large and well-trained model, we can achieve a similar performance compared to those of fine-tuning methods.

[Paper](https://arxiv.org/abs/2005.14165):  Tom B.Brown, Benjamin Mann, Nick Ryder et al. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165). arXiv preprint arXiv:2005.14165

# [Model Architecture](#contents)

GPT3 stacks many layers of decoder of transformer. According to the layer numbers and embedding size, GPT3 has several versions. The largest model contains 96 layers with embedding size of 12288 resulting to a total parameter of 175 billion.

# [Dataset](#contents)

- OpenWebText is utilized as the training data and the training objective is to predict the next token at each position.

# [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash

# run standalone training example

bash scripts/run_standalone_train.sh 0 10 /path/dataset

# run distributed training example

bash scripts/run_distribute_training.sh /path/dataset /path/hccl.json 8

# run evaluation example, now only accuracy and perplexity for lambada and wikitext103 are supported

bash scripts/run_evaluation.sh lambada /your/ckpt /your/data acc

```

For distributed training, an hccl configuration file with JSON format needs to be created in advance.
Please follow the instructions in the link below:
https:gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```shell
.
└─gpt
  ├─README.md
  ├─scripts
    ├─run_standalone_train.sh                 # shell script for standalone training on ascend
    ├─run_distribut_train.sh                  # shell script for distributed training on ascend
    └─run_evaluation.sh                       # shell script for evaluation of ascend
  ├─src
    ├─gpt_wrapper.py                          # backbone code of network
    ├─gpt.py                                  # backbone code of network
    ├─dataset.py                              # data preprocessing
    ├─inference.py                            # evaluation function
    ├─utils.py                                # util function
  ├─train.py                                  # train net for training phase
  └─eval.py                                   # eval net for evaluation
```

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
