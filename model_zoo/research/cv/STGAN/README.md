# Contents

- [Contents](#contents)
    - [STGAN Description](#stgan-description)
    - [Model Architecture](#model-architecture)
    - [Dataset](#dataset)
    - [Environment Requirements](#environment-requirements)
    - [Quick Start](#quick-start)
    - [Script Description](#script-description)
        - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)
            - [Training](#training)
        - [Evaluation Process](#evaluation-process)
            - [Evaluation](#evaluation)
    - [Model Description](#model-description)
    - [Description of Random Situation](#description-of-random-situation)
    - [ModelZoo Homepage](#modelzoo-homepage)

## [STGAN Description](#contents)

STGAN was proposed in CVPR 2019, one of the facial attributes transfer networks using Generative Adversarial Networks (GANs). It introduces a new Selective Transfer Unit (STU) to get better facial attributes transfer than others.

[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_STGAN_A_Unified_Selective_Transfer_Network_for_Arbitrary_Image_Attribute_CVPR_2019_paper.pdf): Liu M, Ding Y, Xia M, et al. STGAN: A Unified Selective Transfer Network for Arbitrary Image
Attribute Editing[C]. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
IEEE, 2019: 3668-3677.

## [Model Architecture](#contents)

STGAN composition consists of Generator, Discriminator and Selective Transfer Unit. Using Selective Transfer Unit can help networks keep more attributes in the long term of training.

## [Dataset](#contents)

In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used: [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

- Dataset size：1011M，202,599 128*128 colorful images, marked as 40 attributes
    - Train：182,599 images
    - Test：18,800 images
- Data format：binary files
    - Note：Data will be processed in celeba.py
- Download the dataset, the directory structure is as follows:

```bash
├── dataroot
    ├── anno
        ├── list_attr_celeba.txt
    ├── image
        ├── 000001.jpg
        ├── ...
```

## [Environment Requirements](#contents)

- Hardware（Ascend）
    - Prepare hardware environment with Ascend processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorial/training/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

## [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```python
# enter script dir, train STGAN
sh scripts/run_standalone_train.sh [DATA_PATH] [EXPERIMENT_NAME] [DEVICE_ID]
# distributed training
sh scripts/run_distribute_train.sh [RANK_TABLE_FILE] [EXPERIMENT_NAME] [DATA_PATH]
# enter script dir, evaluate STGAN
sh scripts/run_eval.sh [DATA_PATH] [EXPERIMENT_NAME] [DEVICE_ID] [CHECKPOINT_PATH]
```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```bash
├── cv
    ├── STGAN
        ├── README.md                    // descriptions about STGAN
        ├── requirements.txt             // package needed
        ├── scripts
        │   ├──run_standalone_train.sh             // train in ascend
        │   ├──run_eval.sh             //  evaluate in ascend
        │   ├──run_distribute_train.sh             // distributed train in ascend
        ├── src
            ├── dataset
                ├── datasets.py                 // creating dataset
                ├── celeba.py                   // processing celeba dataset
                ├── distributed_sampler.py      // distributed sampler
            ├── models
                ├── base_model.py
                ├── losses.py                   // loss models
                ├── networks.py                 // basic models of STGAN
                ├── stgan.py                    // executing procedure
            ├── utils
                ├── args.py                     // argument parser
                ├── tools.py                    // simple tools
        ├── train.py               // training script
        ├── eval.py               //  evaluation script
        ├── export.py               //  model-export script
```

### [Script Parameters](#contents)

```python
Major parameters in train.py and utils/args.py as follows:

--dataroot: The relative path from the current path to the train and evaluation datasets.
--n_epochs: Total training epochs.
--batch_size: Training batch size.
--image_size: Image size used as input to the model.
--device_target: Device where the code will be implemented. Optional value is "Ascend".
```

### [Training Process](#contents)

#### Training

- running on Ascend

  ```bash
  python train.py --dataroot ./dataset --experiment_name 128 > log 2>&1 &
  # or enter script dir, and run the script
  sh scripts/run_standalone_train.sh ./dataset 128 0
  # distributed training
  sh scripts/run_distribute_train.sh ./config/rank_table_8pcs.json 128 /data/dataset
  ```

  After training, the loss value will be achieved as follows:

  ```bash
  # grep "loss is " log
  epoch: 1 step: 1, loss is 2.2791853
  ...
  epoch: 1 step: 1536, loss is 1.9366643
  epoch: 1 step: 1537, loss is 1.6983616
  epoch: 1 step: 1538, loss is 1.0221305
  ...
  ```

  The model checkpoint will be saved in the output directory.

### [Evaluation Process](#contents)

#### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

- running on Ascend

  ```bash
  python eval.py --dataroot ./dataset --experiment_name 128 > eval_log.txt 2>&1 &
  # or enter script dir, and run the script
  sh scripts/run_eval.sh ./dataset 128 0 ./ckpt/generator.ckpt
  ```

  You can view the results in the output directory, which contains a batch of result sample images.

### Model Export

```shell
python export.py --ckpt_path [CHECKPOINT_PATH] --platform [PLATFORM] --file_format[EXPORT_FORMAT]
```

`EXPORT_FORMAT` should be "MINDIR"

## Model Description

### Performance

#### Evaluation Performance

| Parameters                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| Model Version              | V1                                                          |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory, 755G             |
| uploaded Date              | 05/07/2021 (month/day/year)                                 |
| MindSpore Version          | 1.2.0                                                       |
| Dataset                    | CelebA                                                      |
| Training Parameters        | epoch=100,  batch_size = 128                                |
| Optimizer                  | Adam                                                        |
| Loss Function              | Loss                                                        |
| Output                     | predict class                                               |
| Loss                       | 6.5523                                                      |
| Speed                      | 1pc: 400 ms/step;  8pcs:  143 ms/step                       |
| Total time                 | 1pc: 41:36:07                                               |
| Checkpoint for Fine tuning | 170.55M(.ckpt file)                                         |
| Scripts                    | [STGAN script](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/STGAN) |

## [Model Description](#contents)

## [Description of Random Situation](#contents)

In dataset.py, we set the seed inside ```create_dataset``` function.

## [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).
