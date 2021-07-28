# Contents

- [Contents](#contents)
    - [TBNet Description](#tbnet-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
    - [Model Description](#model-description)
        - [Performance](#performance)
            - [Training Performance](#training-performance)
            - [Evaluation Performance](#evaluation-performance)
            - [Inference and Explanation Performance](#inference-explanation-performance)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

# [TBNet Description](#contents)

TB-Net is a knowledge graph based explainable recommender system.

Paper: Shendi Wang, Haoyang Li, Xiao-Hui Li, Caleb Chen Cao, Lei Chen. Tower Bridge Net (TB-Net): Bidirectional Knowledge Graph Aware Embedding Propagation for Explainable Recommender Systems

# [Model Architecture](#contents)

TB-Net constructs subgraphs in knowledge graph based on the interaction between users and items as well as the feature of items, and then calculates paths in the graphs using bidirectional conduction algorithm. Finally we can obtain explainable recommendation results.

# [Dataset](#contents)

[Interaction of users and games](https://www.kaggle.com/tamber/steam-video-games), and the [games' feature data](https://www.kaggle.com/nikdavis/steam-store-games?select=steam.csv) on the game platform Steam are public on Kaggle.

Dataset directory: `./data/{DATASET}/`, e.g. `./data/steam/`.

- train: train.csv, evaluation: test.csv

Each line indicates a \<user\>, an \<item\>, the user-item \<rating\> (1 or 0), and PER_ITEM_NUM_PATHS paths between the item and the user's \<hist_item\> (\<hist_item\> is the item whose the user-item \<rating\> in historical data is 1).

```text
#format:user,item,rating,relation1,entity,relation2,hist_item,relation1,entity,relation2,hist_item,...,relation1,entity,relation2,hist_item  # module [relation1,entity,relation2,hist_item] repeats PER_ITEM_NUM_PATHS times
```

- infer and explain: infer.csv

Each line indicates the \<user\> and \<item\> to be inferred, \<rating\>, and PER_ITEM_NUM_PATHS paths between the item and the user's \<hist_item\> (\<hist_item\> is the item whose the user-item \<rating\> in historical data is 1).
Note that the \<item\> needs to traverse candidate items (all items by default) in the dataset. \<rating\> can be randomly assigned (all values are assigned to 0 by default) and is not used in the inference and explanation phases.

```text
#format:user,item,rating,relation1,entity,relation2,hist_item,relation1,entity,relation2,hist_item,...,relation1,entity,relation2,hist_item  # module [relation1,entity,relation2,hist_item] repeats PER_ITEM_NUM_PATHS times
```

# [Environment Requirements](#contents)

- Hardware（GPU）
    - Prepare hardware environment with GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- Data preprocessing

Process the data to the format in chapter [Dataset](#Dataset) (e.g. 'steam' dataset), and then run code as follows.

- Training

```bash
python train.py \
  --dataset [DATASET] \
  --epochs [EPOCHS]
```

Example:

```bash
python train.py \
  --dataset steam \
  --epochs 20
```

- Evaluation

```bash
python eval.py \
  --dataset [DATASET] \
  --checkpoint_id [CHECKPOINT_ID]
```

Argument `--checkpoint_id` is required.

Example:

```bash
python eval.py \
  --dataset steam \
  --checkpoint_id 8
```

- Inference and Explanation

```bash
python infer.py \
  --dataset [DATASET] \
  --checkpoint_id [CHECKPOINT_ID] \
  --user [USER] \
  --items [ITEMS] \
  --explanations [EXPLANATIONS]
```

Arguments `--checkpoint_id` and `--user` are required.

Example:

```bash
python infer.py \
  --dataset steam \
  --checkpoint_id 8 \
  --user 1 \
  --items 1 \
  --explanations 3
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
.
└─tbnet
  ├─README.md
  ├─data
    ├─steam
        ├─config.json               # data and training parameter configuration
        ├─infer.csv                 # inference and explanation dataset
        ├─test.csv                  # evaluation dataset
        ├─train.csv                 # training dataset
        └─trainslate.json           # explanation configuration
  ├─src
    ├─aggregator.py                 # inference result aggregation
    ├─config.py                     # parsing parameter configuration
    ├─dataset.py                    # generate dataset
    ├─embedding.py                  # 3-dim embedding matrix initialization
    ├─metrics.py                    # model metrics
    ├─steam.py                      # 'steam' dataset text explainer
    └─tbnet.py                      # TB-Net model
  ├─eval.py                         # evaluation
  ├─infer.py                        # inference and explanation
  └─train.py                        # training
```

## [Script Parameters](#contents)

- train.py parameters

```text
--dataset         'steam' dataset is supported currently
--train_csv       the train csv datafile inside the dataset folder
--test_csv        the test csv datafile inside the dataset folder
--device_id       device id
--epochs          number of training epochs
--device_target   run code on GPU
--run_mode        run code by GRAPH mode or PYNATIVE mode
```

- eval.py parameters

```text
--dataset         'steam' dataset is supported currently
--csv             the csv datafile inside the dataset folder (e.g. test.csv)
--checkpoint_id   use which checkpoint(.ckpt) file to eval
--device_id       device id
--device_target   run code on GPU
--run_mode        run code by GRAPH mode or PYNATIVE mode
```

- infer.py parameters

```text
--dataset         'steam' dataset is supported currently
--csv             the csv datafile inside the dataset folder (e.g. infer.csv)
--checkpoint_id   use which checkpoint(.ckpt) file to infer
--user            id of the user to be recommended to
--items           no. of items to be recommended
--reasons         no. of recommendation reasons to be shown
--device_id       device id
--device_target   run code on GPU
--run_mode        run code by GRAPH mode or PYNATIVE mode
```

# [Model Description](#contents)

## [Performance](#contents)

### Training Performance

| Parameters                 | GPU                                                         |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | TB-Net                                                      |
| Resource                   |Tesla V100-SXM2-32GB                                         |
| Uploaded Date              | 2021-08-01                                                  |
| MindSpore Version          | 1.3.0                                                       |
| Dataset                    | steam                                                       |
| Training Parameter         | epoch=20, batch_size=1024, lr=0.001                         |
| Optimizer                  | Adam                                                        |
| Loss Function              | Sigmoid Cross Entropy                                       |
| Outputs                    | AUC=0.8596，Accuracy=0.7761                                 |
| Loss                       | 0.57                                                        |
| Speed                      | 1pc: 90ms/step                                              |
| Total Time                 | 1pc: 297s                                                   |
| Checkpoint for Fine Tuning | 104.66M (.ckpt file)                                        |
| Scripts                    | [TB-Net scripts](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/tbnet) |

### Evaluation Performance

| Parameters                | GPU                           |
| ------------------------- | ----------------------------- |
| Model Version             | TB-Net                        |
| Resource                  | Tesla V100-SXM2-32GB          |
| Uploaded Date             | 2021-08-01                    |
| MindSpore Version         | 1.3.0                         |
| Dataset                   | steam                         |
| Batch Size                | 1024                          |
| Outputs                   | AUC=0.8252，Accuracy=0.7503   |
| Total Time                | 1pc: 5.7s                     |

### Inference and Explanation Performance

| Parameters                | GPU                                   |
| --------------------------| ------------------------------------- |
| Model Version             | TB-Net                                |
| Resource                  | Tesla V100-SXM2-32GB                  |
| Uploaded Date             | 2021-08-01                            |
| MindSpore Version         | 1.3.0                                 |
| Dataset                   | steam                                 |
| Outputs                   | Recommendation Result and Explanation |
| Total Time                | 1pc: 3.66s                            |

# [Description of Random Situation](#contents)

- Initialization of embedding matrix in `tbnet.py` and `embedding.py`.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).