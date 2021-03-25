# 目录

<!-- TOC -->

- [目录](#目录)
- [GPT-2模型](#GPT-2模型)
- [模型架构](#模型架构)
- [下游任务](#下游任务)
    - [脚本说明](#脚本说明)
    - [模型转换](#模型转换)
    - [准备数据集](#准备数据集)
        - [Language Modeling 语言建模任务](#Language Modeling语言建模任务)
        - [Children's Book Test 任务](#Children's Book Test任务)
        - [LAMBADA 任务](#LAMBADA任务)
        - [Reading Comprehension 任务](#Reading Comprehension任务)
        - [Summarization 任务](#Summarization任务)
        - [Translation 任务](#Translation任务)
    - [配置](#配置)
    - [微调&评估过程](#微调&训练评估过程)
        - [Language Modeling 任务](#Language Modeling任务)
            - 微调
            - 评估
        - [Children's Book Test 任务](#Children's Book Test任务)
            - 评估
        - [LAMBADA 任务](#LAMBADA任务)
            - 评估
        - [Reading Comprehension 任务](#Reading Comprehension任务)
            - 评估
        - [Summarization 任务](#Summarization任务)
            - 评估
        - [Translation 任务](#Translation任务)
            - 评估
- [环境要求](#环境要求)
    - [平台](#平台)
    - [其他要求](#其他要求)
- [性能](#性能)
    - [推理性能](#推理性能)
        - [Language Modeling 任务](#Language Modeling任务)
        - [Children's Book Test 任务](#Children's Book Test任务)
        - [LAMBADA 任务](#LAMBADA任务)
        - [Reading Comprehension 任务](#Reading Comprehension任务)
        - [Summarization 任务](#Summarization任务)
        - [Translation 任务](#Translation任务)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [其他](#其他)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# GPT-2模型

[GPT-2介绍](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) 由Open于2019年发布。GPT-2模型是继承于GPT模型，GPT-2是一个非常庞大的语言模型，它主要是用于预测下一个单词。按照参数量的大小，GPT-2模型可分为small（117M）、medium（345M）、large（762M）、xlarge（1542M）。

[GPT-2介绍](https://openai.com/blog/better-language-models/)

[GPT-2论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf): Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.

# 模型架构

GPT-2模型由Transformer的解码器实现，Transformer包括多个编码器层和多个解码器层，但在GPT-2模型中仅使用了Transformer的解码器部分。
微调时，根据不同的任务，采用不同的数据集对预训练的模型进行微调。
测试过程中，通过微调后的模型预测结果，对于某些任务可以直接进行zero-shot评估即可。

# 下游任务

本文主要涉及6个下游任务，包括：

- Language Modeling 任务
- Children‘s Book Test 任务
- LAMBADA任务
- Reading Comprehension任务
- Summarization任务
- Translation任务

数据集相关信息，参见[https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)。

## 脚本说明

GPT-2脚本及代码结构如下：

```text
├── GPT-2
  ├── README.md                              // MASS模型介绍
  ├── scripts
  │   ├──run_cbt.sh                          // CBT任务的微调&评估脚本
  │   ├──run_lambada.sh                      // LAMBADA任务的微调&评估脚本
  │   ├──run_language_model.sh               // 语言建模任务的微调&评估脚本
  │   ├──run_read_comprehension.sh           // 阅读理解任务的微调&评估脚本
  │   ├──run_summarization.sh                // 摘要生成任务的微调&评估脚本
  │   ├──run_translation.sh                  // 翻译任务的微调&评估脚本
  ├──src
  │   ├──clip_grad_utils.py                  // 用于梯度裁剪
  |   ├──dataset.py                          // 数据集加载用于微调或推理
  │   ├──finetune_eval_config.py             // 微调和推理配置文件
  │   ├──gpt2_for_finetune.py                // 用于梯度裁剪
  |   ├──GPT2_generation.py                  // 生成模块
  │   ├──GPT2_model.py                       // GPT2模型脚本
  │   ├──GPT2ForCBT.py                       // CBT任务的模型脚本
  │   ├──GPT2ForLanguageModel.py             // 语言建模任务的模型脚本
  │   ├──GPT2ForReadComprehension.py         // 阅读理解任务的模型脚本
  │   ├──GPT2ForSummarization.py             // 摘要生成任务的模型脚本
  │   ├──GPT2ForTranslation.py               // 翻译任务的模型脚本
  │   ├──weight_init.py                      // 初始化权重
  │   ├──utils
  │      ├──bleu_score.py                    // 用于计算BLEU分数
  │      ├──rouge_score.py                   // 用于计算ROUGE分数
  │      ├──CrossEntropy.py                  // 交叉熵损失
  │      ├──data_preprocess.py               // 数据集预处理脚本
  │      ├──generation_utils.py              // 用于帮助生成模型，包含采样等方法
  │      ├──get_config_setting.py            // 获取配置信息
  │      ├──task_utils.py                    // 辅助下游任务的功能脚本
  │      ├──lr_schedule.py                   // 学习率策略脚本
  │      ├──metric_method.py                 // 下游任务的评价指标
  │      ├──tensor_manipulations.py          // 涉及张量操作
  │      ├──tokenization.py                  // 标记化，包含BPE编码和解码
  │      ├──pretrain-data
  │          ├──stopwords.txt                // 用于LAMBADA任务的stopword filter
  ├──create_cbt_data.py                      // 用于CBT任务创建mindrecord
  ├──create_lambada_data.py                  // 用于lambada任务创建mindrecord
  ├──create_lambada_data.py                  // 用于其他任务创建mindrecord
  ├──create_summary_data.py                  // 用于summarization任务创建mindrecord
  ├──download_cnn_dailymail.py               // 下载CNN & Dailymail数据集
  ├──cnn_dataset_sampler.py                  // CNN & Dailymail训练集采样器
  ├──eval_rc_addition_answer.py              // 使用addition_answer评估阅读理解任务
  ├──run_CBT_task.py                         // CBT任务微调&推理API入口
  ├──run_lambada.py                          // LAMBADA任务微调&推理API入口
  ├──run_language_mdoel.py                   // 语言建模任务微调&推理API入口
  ├──run_ReadComprehension.py                // 阅读理解任务微调&推理API入口
  ├──run_summarization.py                    // 摘要生成任务微调&推理API入口
  ├──run_translation.py                      // 翻译任务微调&推理API入口
  ├──task_dataset_preprocess.py              // 各个任务的数据集处理入口
  ├──convert_tf_ckpt
  │      ├──read_weight_tf.py                // 读取tensorflow下的预训练模型
  │      ├──trans_dict.py                    // 模型参数名称字典
  │      ├──save_weight_ms.py                // 生成mindspore ckpt
  ├──third_party
  │      ├──gpt2-merges.txt
  │      ├──gpt2-vocab.json                  // GPT-2预训练词表
  │      ├──bleu.py                          // 辅助bleu值计算的第三方代码


```

## 模型转换

- 下载GPT-2的预训练模型 [GPT-2预训练模型下载](https://github.com/openai/gpt-2/blob/master/download_model.py)

- 在tensorflow的环境下，运行`read_weight_tf.py`，示例代码如下：

`python read_weight_tf.py --ckpt_file_path=/{path}/model.ckpt`

- 在mindspore的环境下，运行`save_weight_ms.py`，示例代码如下：

`python save_weight_ms.py --output_file_name="mindspore_gpt2_small.ckpt"`

## 准备数据集

### Language Modeling语言建模任务

#### WikiText2 、WikiText103、PTB、1BW 数据集

- [WikiText2数据集下载](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip) 解压后使用`wikitext-2 /wiki.test.tokens`作为测试集
- [WikiText103数据集下载](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip) 解压后使用`wikitext-103 /wiki.test.tokens`作为测试集
- [PTB数据集下载](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz) 解压后使用  `/simple-examples/data/ptb.test.txt` 测试集，使用 `/simple-examples/data/ptb.test.txt` 作为训练集
- [1BW数据集下载](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz) 解压后使用`1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050`作为测试集，使用`1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100`作为原始训练集，进行随机采样后得到30000条训练集样本

使用`task_dataset_preprocess.py`可以对以上数据集进行清洗。

`task_dataset_preprocess.py`的主要参数如下：

```bash
--task:          The GPT-2 downstream task, including [LanguageModeling, CBT, Translation, Lambada, Summarization, ReadingComprehension].
--input_file:    The raw dataset path.
--dataset:       The name of dataset which should be processed, only for LanguageModeling task.
--output_file:   The output dataset path after preprocessing.
--condition:     Process train or test dataset, including [train, test], only for 1BW and CNN & DailyMail dataset.
```

示例代码如下：

清洗PTB训练集和测试集

```bash
python task_dataset_preprocess.py --task "LanguageModeling" --input_file /{path}/ptb.test.txt --dataset "ptb" --output_file /{path}/ptb_clean_test.txt --condition "test"
```

使用`create_lm_data.py`可以将以上数据集格式转换为mindrecord

`create_lm_data.py`的主要参数如下：

```bash
--input_file:      Input raw text file.
--output_file:     Output MindRecord file.
--num_splits:      The MindRecord file will be split into the number of partition.
--max_seq_length:  Maximum sequence length.
--vocab_file:      url of gpt2-vocab.json.
--merge_file:      url of gpt2-merges.txt
```

示例代码如下：

```bash
python create_lm_data.py --input_file /{path}/ptb.test.txt --output_file /{path}/ptb-test-mindrecord --num_splits 1 --max_length 1024 --vocab_file={path} --merge_file={path}
```

### Children's Book Test任务

#### CBT-CN / CBT-NE 数据集

- [CBT数据集下载](http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz)  使用在`/data`目录下使用`cbtest_CN_valid_2000ex.txt、cbtest_NE_valid_2000ex.txt`作为该任务的评估集，清洗该数据集，示例代码如下：

```bash
python task_dataset_preprocess.py --task "CBT" --input_file /{path}/cbtest_CN_valid_2000ex.txt --dataset "cbt" --output_file /{path}/cbt_cn_valid.txt
```

使用`create_cbt_data.py`可以将以上数据集格式转换为mindrecord

`create_cbt_data.py`的主要参数如下：

```bash
--input_file:      Input raw text file.
--output_file:     Output MindRecord file.
--num_splits:      The MindRecord file will be split into the number of partition.
--max_seq_length:  Maximum sequence length.
--num_choice:      Number of choices.
--vocab_file:      url of gpt2-vocab.json.
--merge_file:      url of gpt2-merges.txt
```

示例代码如下：

```bash
python create_cbt_data.py --input_file /{path}/ptb.test.txt --output_file /{path}/ptb-test-mindrecord --num_splits 1 --max_length 1024 --num_choice 10 --vocab_file={path} --merge_file={path}
```

### LAMBADA任务

#### LAMBADA 数据集

- [LAMBADA数据集下载](https://zenodo.org/record/2630551#.X-yCSTTithH)  使用`lambada_test_plain_text.txt`作为该任务的评估集，清洗该数据集，示例代码如下：

```bash
python task_dataset_preprocess.py --task "LAMBADA" --input_file /{path}/lambada_test_plain_text.txt --dataset "LAMBADA" --output_file /{path}/lambada_test_clean.txt
```

使用`create_lambada_data.py`可以将以上数据集格式转换为mindrecord

`create_lambada_data.py`的主要参数如下：

```bash
--input_file:      Input raw text file.
--output_file:     Output MindRecord file.
--num_splits:      The MindRecord file will be split into the number of partition.
--max_seq_length:  Maximum sequence length.
--vocab_file:      url of gpt2-vocab.json.
--merge_file:      url of gpt2-merges.txt
```

示例代码如下：

```bash
python create_lambada_data.py --input_file /{path}/lambada_test_clean.txt --output_file /{path}/lambada-test-mindrecord --num_splits 1 --max_length 1024 --vocab_file={path} --merge_file={path}
```

### Reading Comprehension 任务

#### CoQA数据集

- [CoQA数据集下载](http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json)  使用`coqa-dev-v1.0.json`作为该任务的评估集，清洗该数据集，示例代码如下：

```bash
python task_dataset_preprocess.py --task "ReadingComprehension" --input_file /{path}/coqa-dev-v1.0.json --dataset "coqa" --output_file /{path}/coqa_dev.txt
```

使用`create_lm_data.py`可以将以上数据集格式转换为mindrecord

示例代码如下：

```bash
python create_lm_data.py --input_file /{path}/coqa_dev.txt --output_file /{path}/coqa-dev-mindrecord --num_splits 1 --max_length 1024 --vocab_file={path} --merge_file={path}
```

### Summarization 任务

#### CNN & Dailymail数据集

- 下载该数据集，使用`download_cnn_dailymail.py`脚本进行下载，示例代码如下：

```bash
下载测试集
python download_cnn_dailymail.py --dir ./cnn_dailymail/ --split test

下载训练集
python download_cnn_dailymail.py --dir ./cnn_dailymail/ --split train
```

从训练集中随机采用10000条样本作为最终的微调的训练集，使用`cnn_dataset_sampler.py`脚本进行训练的采样操作，生成新的训练集，示例代码如下：

```bash
GPT-2 small和GPT-2 medium模型的训练集中seq_length=1024, 因此该脚本中设置max_length=1022
python cnn_dataset_sampler.py --input_path="/{path}/cnn_train.txt"
                              --output_path="/{path}/cnn_train_hint_small.txt"
                              --replace_hint="true"
                              --sample="true"
                              --max_length=1022
                              --prob=0.25
                              --max_items=10000
                              --hint="TL;DR:"


GPT-2 large模型的训练集中seq_length=768,因此该脚本中设置max_length=766
python cnn_dataset_sampler.py --input_path="/{path}/cnn_train.txt"
                              --output_path="/{path}/cnn_train_hint_large.txt"
                              --replace_hint="true"
                              --sample="true"
                              --max_length=766
                              --prob=0.25
                              --max_items=10000
                              --hint="TL;DR:"
```

使用`create_summary_data.py`可以将以上数据集格式转换为mindrecord

示例代码如下：

```bash
python create_summary_data.py --input_file /{path}/cnn_dailymail_test.txt --output_file /{path}/cnn_dailymail-test-mindrecord --num_splits 1 --max_length 1024 --vocab_file={path} --merge_file={path} --mode 'cnn_dailymail'
```

### Translation 任务

#### WMT14 En-Fr数据集

- [WMT14 En-Fr数据集下载](http://statmt.org/wmt14/test-full.tgz)  使用`newstest2014-fren-ref.en.sgm`和`newstest2014-fren-ref.fr.sgm`作为该任务的评估集，合并且清洗该数据集，示例代码如下：

```bash
python task_dataset_preprocess.py --task "Translation" --input_file /{path}/test-full --dataset "wmt14" --output_file /{path}/wmt14
```

在`output_file`路径下会生成两个文件`wmt14.en_fr.txt`和`wmt14.fr_en.txt`，分别用于评估`En-Fr`和`Fr-En`。

使用`create_lm_data.py`可以将以上数据集格式转换为mindrecord

示例代码如下：

```bash
python create_lm_data.py --input_file /{path}/wmt14.en_fr.txt --output_file /{path}/en-fr-mindrecord --num_splits 1 --max_length 1024 --vocab_file={path} --merge_file={path}

python create_lm_data.py --input_file /{path}/wmt14.fr_en.txt --output_file /{path}/fr-en-mindrecord --num_splits 1 --max_length 1024 --vocab_file={path} --merge_file={path}
```

## 配置

`src/finetune_eval_config.py`为GPT-2模型训练和推理的配置文件，便于为大多数选项及参数赋值，包括GPT-2 模型规模、模型的配置、优化器参数等。
有关属性的详细信息，参见`src/finetune_eval_config.py`文件。

## 微调&评估过程

### Language Modeling 语言建模任务

#### 微调

- PTB数据集

GPT-2 small / GPT-2 medium / GPT-2 large模型需要在PTB训练集上进行微调。微调模型时，只需要使用shell脚本`scripts/run_language_model.sh`即可，脚本中可以设置环境变量，执行`GPT-2`下的`scripts/run_language_model.sh`脚本。

微调模型时，首先配置`src/finetune_eval_config.py`中的选项：

将`cfg`下的`gpt2_network`设置为相应的GPT-2模型大小`[small/medium/large]`。
将`cfg`下的`optimizer`设置为`Lamb`，进行优化器的选择（可采用'momentum/adam/lamb’）。
选定了GPT-2模型后需要设置模型的参数，包括`batch_size`和`seq_length`。

而后执行`scripts/run_language_model.sh`这个shell脚本：

```bash
sh scripts/run_language_model.sh   --device_target="Ascend"
                                   --do_train="true"
                                   --do_eval="false"
                                   --epoch_num=1
                                   --train_data_shuffle="true"
                                   --eval_data_shuffle="false"
                                   --save_finetune_ckpt_path={save_finetune_ckpt_path}
                                   --load_pretrain_ckpt_path={load_pretrain_ckpt_path}
                                   --train_data_file_path={train_data_file_path}
```

日志和输出文件可以在`./ms_log/`路径下获取。

```bash
sh scripts/run_language_model.sh [--options]
```

`run_language_model.sh`的用法如下：

```text
usage: run_language_model.sh   [--device_target DEVICE_TARGET] [--device_id N]
                               [--metric_method METRIC_METHOD]
                               [--do_train DO_TRAIN] [--do_eval DO_EVAL]
                               [--eval_type EVAL_TYPE] [--epoch_num N]
                               [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                               [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                               [--save_finetune_ckpt_path SAVE_FINETUNE_CKPT_PATH]
                               [--load_pretrain_ckpt_path LOAD_PRETRAIN_CKPT_PATH]
                               [--load_finetune_ckpt_path LOAD_FINETUNE_CKPT_PATH]
                               [--train_data_file_path TRAIN_DATA_FILE_PATH]
                               [--eval_data_file_path EVAL_DATA_FILE_PATH]
options:
    --device_target                   Device type. Default: "Ascend"
    --device_id                       ID of target device
    --metric_method                   The eval method including [PPL]. Default: "PPL"
    --do_train                        Enable train. Default: "false"
    --do_eval                         Enable evaluation. Default: "true"
    --eval_type                       The type of evaluation including [zero-shot, finetuned]. Default: "zero-shot"
    --epoch_num                       Epoch number. Default: 1
    --train_data_shuffle              Enable train data shuffle. Default: "true"
    --eval_data_shuffle               Enable eval data shuffle. Default: "false"
    --save_finetune_ckpt_path         Save the finetuned checkpoint path
    --load_pretrain_ckpt_path         Load the checkpoint file path for train
    --load_finetune_ckpt_path         Load the checkpoint file path for evaluation
    --train_data_file_path            Data path, it is better to use absolute path
    --eval_data_file_path             Data path, it is better to use absolute path
```

- 1BW数据集

GPT-2 large模型需要在1BW训练集上进行微调。微调模型时，只需要使用shell脚本`run_language_model.sh`即可，脚本中可以设置环境变量，执行`GPT-2`下的`run_language_model.py`脚本。该微调方法与PTB数据集的一致。

#### 评估

GPT-2模型可以在`WikiText2/WikiText103/PTB/1BW`测试集上进行对应的评估，针对以上数据集的评估，其评估方法采用PPL，即设置`--metric_method="PPL"`。

评估模型时，只需要使用shell脚本`run_language_model.sh`即可，脚本中可以设置环境变量，执行`GPT-2`下的`run_language_model.py`脚本。

评估模型时，首先配置`src/finetune_eval_config.py`，而后执行`scripts/run_language_model.sh`这个shell脚本，若该模型在某个数据集上被微调了，则使用该模型进行对应测试集的评估时需要设置`--eval_type="finetuned"`，否则设置`eval_type="zero-shot"`，除此之外`--load_finetune_ckpt_path`是微调好后的checkpoint文件位置

```bash
sh scripts/run_language_model.sh   --device_target="Ascend"
                                   --metric_method="PPL"
                                   --do_train="false"
                                   --do_eval="true"
                                   --eval_type="finetuned"
                                   --train_data_shuffle="true"
                                   --eval_data_shuffle="false"
                                   --load_finetune_ckpt_path={load_eval_ckpt_path}
                                   --eval_data_file_path={eval_data_file_path}
```

日志和输出文件可以在`./ms_log/`路径下获取。

### Children's Book Test任务

#### 评估

GPT-2模型可以在`CBT-CN/CBT-NE`验证集上进行对应的评估，针对以上数据集的评估，其评估方法采用Accuracy，即设置`--metric_method="Accuracy"`。

评估模型时，只需要使用shell脚本`run_cbt.sh`即可，脚本中可以设置环境变量，执行`GPT-2`下的`run_CBT_task.py`脚本。

评估模型时，首先配置`src/finetune_eval_config.py`，而后执行`scripts/run_cbt.sh`这个shell脚本，且设置`eval_type="zero-shot"`，除此之外`--load_finetune_ckpt_path`是只需加载预训练好的checkpoint文件

```bash
sh scripts/run_cbt.sh   --device_target="Ascend"
                        --num_choice=10
                        --metric_method="Accuarcy"
                        --do_train="false"
                        --do_eval="true"
                        --eval_type="zero-shot"
                        --train_data_shuffle="true"
                        --eval_data_shuffle="false"
                        --load_finetune_ckpt_path={load_eval_ckpt_path}
                        --eval_data_file_path={eval_data_file_path}
```

日志和输出文件可以在`./ms_log/`路径下获取。

```bash
sh scripts/run_cbt.sh [--options]
```

`run_cbt.sh`的用法如下：

```text
usage: run_CBT_task.sh   [--device_target DEVICE_TARGET] [--device_id N][--num_choice N]
                         [--metric_method METRIC_METHOD]
                         [--do_train DO_TRAIN] [--do_eval DO_EVAL]
                         [--eval_type EVAL_TYPE] [--epoch_num N]
                         [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                         [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                         [--save_finetune_ckpt_path SAVE_FINETUNE_CKPT_PATH]
                         [--load_pretrain_ckpt_path LOAD_PRETRAIN_CKPT_PATH]
                         [--load_finetune_ckpt_path LOAD_FINETUNE_CKPT_PATH]
                         [--train_data_file_path TRAIN_DATA_FILE_PATH]
                         [--eval_data_file_path EVAL_DATA_FILE_PATH]
options:
    --device_target                   Device type. Default: "Ascend"
    --device_id                       ID of target device
    --num_choice                      The number of choice in CBT task
    --metric_method                   The eval method including [Accuracy]. Default: "Accuracy"
    --do_train                        Enable train. Default: "false"
    --do_eval                         Enable evaluation. Default: "true"
    --eval_type                       The type of evaluation including [zero-shot, finetuned]. Default: "zero-shot"
    --epoch_num                       Epoch number. Default: 1
    --train_data_shuffle              Enable train data shuffle. Default: "true"
    --eval_data_shuffle               Enable eval data shuffle. Default: "false"
    --save_finetune_ckpt_path         Save the finetuned checkpoint path
    --load_pretrain_ckpt_path         Load the checkpoint file path for train
    --load_finetune_ckpt_path         Load the checkpoint file path for evaluation
    --train_data_file_path            Data path, it is better to use absolute path
    --eval_data_file_path             Data path, it is better to use absolute path

```

### LAMBADA任务

#### 评估

GPT-2模型可以在`LAMBADA`测试集上进行对应的评估，针对以上数据集的评估，其评估方法采用Accuracy和PPL，即设置`--metric_method="Accuracy"` 或者`--metric_method="PPL"`。

评估模型时，只需要使用shell脚本`run_lambada.sh`即可，脚本中可以设置环境变量，执行`GPT-2`下的`run_lambada.py`脚本。

评估模型时，首先配置`src/finetune_eval_config.py`，而后执行`scripts/run_lambada.sh`这个shell脚本，且设置`eval_type="zero-shot"`，除此之外`--load_finetune_ckpt_path`是只需加载预训练好的checkpoint文件

评估Accuracy

```bash
sh scripts/run_lambada.sh   --device_target="Ascend"
                            --metric_method="Accuarcy"
                            --do_train="false"
                            --do_eval="true"
                            --eval_type="zero-shot"
                            --train_data_shuffle="true"
                            --eval_data_shuffle="false"
                            --generate_length_dynamically="true"
                            --load_finetune_ckpt_path={load_eval_ckpt_path}
                            --eval_data_file_path={eval_data_file_path}
                            --tokenizer_file_path={tokenizer_file_path}
                            --stop_word_file_path={stop_word_file_path}
```

评估PPL

```bash
sh scripts/run_lambada.sh   --device_target="Ascend"
                            --metric_method="PPL"
                            --do_train="false"
                            --do_eval="true"
                            --eval_type="zero-shot"
                            --train_data_shuffle="true"
                            --eval_data_shuffle="false"
                            --load_finetune_ckpt_path={load_eval_ckpt_path}
                            --eval_data_file_path={eval_data_file_path}
```

日志和输出文件可以在`./ms_log/`路径下获取。

```bash
sh scripts/run_lambada.sh [--options]
```

```text
usage: run_lambada.sh   [--device_target DEVICE_TARGET] [--device_id N]
                        [--metric_method METRIC_METHOD]
                        [--do_train DO_TRAIN] [--do_eval DO_EVAL]
                        [--eval_type EVAL_TYPE] [--epoch_num N]
                        [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                        [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                        [--generate_length_dynamically GENERATE_LENGTH_DYNAMICALLY]
                        [--save_finetune_ckpt_path SAVE_FINETUNE_CKPT_PATH]
                        [--load_pretrain_ckpt_path LOAD_PRETRAIN_CKPT_PATH]
                        [--load_finetune_ckpt_path LOAD_FINETUNE_CKPT_PATH]
                        [--train_data_file_path TRAIN_DATA_FILE_PATH]
                        [--eval_data_file_path EVAL_DATA_FILE_PATH]
                        [--tokenizer_file_path TOKENIZER_FILE_PATH]
                        [--stop_word_file_path STOP_WORD_FILE_PATH]
options:
    --device_target                   Device type. Default: "Ascend"
    --device_id                       ID of target device
    --metric_method                   The eval method including [Accuracy, PPL]. Default: "Accuracy"
    --do_train                        Enable train. Default: "false"
    --do_eval                         Enable evaluation. Default: "true"
    --eval_type                       The type of evaluation including [zero-shot, finetuned]. Default: "zero-shot"
    --epoch_num                       Epoch number. Default: 1
    --train_data_shuffle              Enable train data shuffle. Default: "true"
    --eval_data_shuffle               Enable eval data shuffle. Default: "false"
    --generate_length_dynamically     Enable generate_length_Dynamically. Default: "true"
    --save_finetune_ckpt_path         Save the checkpoint path
    --load_pretrain_ckpt_path         Load the checkpoint file path
    --load_finetune_ckpt_path         Load the checkpoint file path
    --train_data_file_path            Data path, it is better to use absolute path
    --eval_data_file_path             Data path, it is better to use absolute path
    --tokenizer_file_path             pretrained vocab and merge file path
    --stop_word_file_path             The stop word file path
```

### Reading Comprehension任务

#### 评估

GPT-2模型可以在`CoQA`开发集上进行对应的评估，针对以上数据集的评估，其评估方法采用F1，即设置`--metric_method="F1"` 。

评估模型时，只需要使用shell脚本`run_read_comprehension.sh`即可，脚本中可以设置环境变量，执行`GPT-2`下的`run_read_comprehension.py`脚本。

评估模型时，首先配置`src/finetune_eval_config.py`，而后执行`scripts/run_read_comprehension.sh`这个shell脚本，且设置`eval_type="zero-shot"`，除此之外`--load_finetune_ckpt_path`是只需加载预训练好的checkpoint文件

```bash
sh scripts/run_read_comprehension.sh   --device_target="Ascend"
                                       --metric_method="F1"
                                       --do_train="false"
                                       --do_eval="true"
                                       --eval_type="zero-shot"
                                       --train_data_shuffle="true"
                                       --eval_data_shuffle="false"
                                       --load_finetune_ckpt_path={load_eval_ckpt_path}
                                       --eval_data_file_path={eval_data_file_path}
                                       --tokenizer_file_path={tokenizer_file_path}
                                       --generate_length=55
                                       --top_k=1
                                       --top_p="1.0"
                                       --temperature="1.0"
```

日志和输出文件可以在`./ms_log/`路径下获取。而后将得到的日志文件作为`eval_rc_addition_answer.py`脚本的`input_file`，同时将原CoQA开发集`coqa-dev-v1.0.json`作为`addition_file`。

执行`python eval_rc_addition_answer.py --input_file={path} --addition_file={path}`得到最终的F1值。

```bash
sh scripts/run_read_comprehension.sh [--options]
```

```text
usage: run_read_comprehension.sh   [--device_target DEVICE_TARGET] [--device_id N]
                                   [--metric_method METRIC_METHOD]
                                   [--do_train DO_TRAIN] [--do_eval DO_EVAL]
                                   [--eval_type EVAL_TYPE] [--epoch_num N]
                                   [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                                   [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                                   [--save_finetune_ckpt_path SAVE_FINETUNE_CKPT_PATH]
                                   [--load_pretrain_ckpt_path LOAD_PRETRAIN_CKPT_PATH]
                                   [--load_finetune_ckpt_path LOAD_FINETUNE_CKPT_PATH]
                                   [--train_data_file_path TRAIN_DATA_FILE_PATH]
                                   [--eval_data_file_path EVAL_DATA_FILE_PATH]
                                   [--tokenizer_file_path TOKENIZER_FILE_PATH]
                                   [--generate_length N] [--top_k N] [--top_p TOP_P]
                                   [--temperature TEMPERATURE]
options:
    --device_target                   Device type. Default: "Ascend"
    --device_id                       ID of target device
    --metric_method                   The eval method including [F1]. Default: "F1"
    --do_train                        Enable train. Default: "false"
    --do_eval                         Enable evaluation. Default: "false"
    --eval_type                       The type of evaluation including [zero-shot, finetuned]. Default: "zero-shot"
    --epoch_num                       Epoch number. Default: 1
    --train_data_shuffle              Enable train data shuffle. Default: "true"
    --eval_data_shuffle               Enable eval data shuffle. Default: "false"
    --save_finetune_ckpt_path         Save the checkpoint path
    --load_pretrain_ckpt_path         Load the checkpoint file path
    --load_finetune_ckpt_path         Load the checkpoint file path
    --train_data_file_path            Data path, it is better to use absolute path
    --eval_data_file_path             Data path, it is better to use absolute path
    --tokenizer_file_path             pretrained vocab and merge file path
    --generate_length                 The generation length of answer sentence
    --top_k                           Parameter for Top-K sampling
    --top_p                           Parameter for Top-P sampling
    --temperature                     Parameter for generation, greater if generation more diverse
```

### Summarization任务

#### 评估

GPT-2模型可以在`CNN_Dailymail`开发集上进行对应的评估，针对以上数据集的评估，其评估方法采用F1，即设置`--metric_method="ROUGE"` 。

评估模型时，只需要使用shell脚本`run_summarization.sh`即可，脚本中可以设置环境变量，执行`GPT-2`下的`run_summarization.py`脚本。

评估模型时，首先配置`src/finetune_eval_config.py`，而后执行`scripts/run_summarization.sh`这个shell脚本，且对于`hint`的情况设置`eval_type="finetuned"`，`--load_finetune_ckpt_path`是需要加载微调好的checkpoint文件；而对于`no hint`的情况设置`eval_type="zero-shot"`除此之外`--load_finetune_ckpt_path`是只需加载预训练好的checkpoint文件

```bash
sh scripts/run_summarization.sh   --device_target="Ascend"
                                  --do_train="false"
                                  --do_eval="true"
                                  --metric_method="Rouge"
                                  --train_data_shuffle="true"
                                  --eval_data_shuffle="false"
                                  --generate_length=100
                                  --top_k=2
                                  --top_p="1.0"
                                  --temperature="1.0"
                                  --eval_type="finetuned"
                                  --load_finetune_ckpt_path={load_eval_ckpt_path}
                                  --eval_data_file_path={eval_data_file_path}
                                  --tokenizer_file_path={tokenizer_file_path}

```

日志和输出文件可以在`./ms_log/`路径下获取。

```bash
sh scripts/run_summarization.sh [--options]
```

`run_summarization.sh`的用法如下：

```text
usage: run_summarization.sh   [--device_target DEVICE_TARGET] [--device_id N][--num_choice N]
                              [--metric_method METRIC_METHOD]
                              [--do_train DO_TRAIN] [--do_eval DO_EVAL]
                              [--eval_type EVAL_TYPE] [--epoch_num N]
                              [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                              [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                              [--save_finetune_ckpt_path SAVE_FINETUNE_CKPT_PATH]
                              [--load_pretrain_ckpt_path LOAD_PRETRAIN_CKPT_PATH]
                              [--load_finetune_ckpt_path LOAD_FINETUNE_CKPT_PATH]
                              [--train_data_file_path TRAIN_DATA_FILE_PATH]
                              [--eval_data_file_path EVAL_DATA_FILE_PATH]
options:
    --device_target                   Device type. Default: "Ascend"
    --device_id                       ID of target device
    --do_train                        Enable train. Default: false.
    --do_eval                         Enable evaluation. Default: false.
    --metric_method                   The eval method including [Rouge(Rouge1,Rouge2,RougeL,Rouge Avg)]. Default: Rouge. Default: "false"
    --epoch_num                       Epoch number. Default: 2.
    --train_data_shuffle              Enable train data shuffle. Default: true.
    --eval_data_shuffle               Enable eval data shuffle. Default: false.
    --save_finetune_ckpt_path         Save the checkpoint path.
    --load_pretrain_ckpt_path         Load the checkpoint file path.
    --load_finetune_ckpt_path         Load the checkpoint file path.
    --train_data_file_path            Data path, it is better to use absolute path.
    --eval_data_file_path             Data path, it is better to use absolute path.
    --eval_type                       The type of evaluation including [zero-shot, finetuned]. Default: zero-shot.
    --top_k                           Top k tokens chosen for sampling.
    --top_p                           Top p accumulated probability threshold for logit to be counted.
    --generate_length                 The number of generated tokens.
    --temperature                     Temperature on logits for sampling.
    --tokenizer_file_path             Vocab & merge file path.
```

### Translation任务

#### 评估

GPT-2模型可以在`WMT14 En-Fr`和`WMT14 Fr-En`测试集上进行对应的评估，针对以上数据集的评估，其评估方法采用BLEU，即设置`--metric_method="BLEU"` 。

注：读者需要自行下载`bleu.py`脚本[脚本链接](https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py), 而后将该脚本放置于`src/utils/`目录下

评估模型时，只需要使用shell脚本`run_translation.sh`即可，脚本中可以设置环境变量，执行`GPT-2`下的`run_translation.py`脚本。

评估模型时，首先配置`src/finetune_eval_config.py`，而后执行`scripts/run_translation.sh`这个shell脚本，且设置`eval_type="zero-shot"`，除此之外`--load_finetune_ckpt_path`是只需加载预训练好的checkpoint文件

```bash
sh scripts/run_translation.sh   --device_target="Ascend"
                                --metric_method="BLEU"
                                --do_train="false"
                                --do_eval="true"
                                --eval_type="zero-shot"
                                --train_data_shuffle="true"
                                --eval_data_shuffle="false"
                                --load_finetune_ckpt_path={load_eval_ckpt_path}
                                --eval_data_file_path={eval_data_file_path}
                                --tokenizer_file_path={tokenizer_file_path}
                                --generate_length=100
                                --top_k=1
                                --top_p="1.0"
                                --temperature="1.0"
```

```bash
sh scripts/run_translation.sh [--options]
```

```text
usage: run_translation.sh   [--device_target DEVICE_TARGET] [--device_id N]
                            [--metric_method METRIC_METHOD]
                            [--do_train DO_TRAIN] [--do_eval DO_EVAL]
                            [--eval_type EVAL_TYPE] [--epoch_num N]
                            [--train_data_shuffle TRAIN_DATA_SHUFFLE]
                            [--eval_data_shuffle EVAL_DATA_SHUFFLE]
                            [--save_finetune_ckpt_path SAVE_FINETUNE_CKPT_PATH]
                            [--load_pretrain_ckpt_path LOAD_PRETRAIN_CKPT_PATH]
                            [--load_finetune_ckpt_path LOAD_FINETUNE_CKPT_PATH]
                            [--train_data_file_path TRAIN_DATA_FILE_PATH]
                            [--eval_data_file_path EVAL_DATA_FILE_PATH]
                            [--tokenizer_file_path TOKENIZER_FILE_PATH]
                            [--generate_length N] [--top_k N] [--top_p TOP_P]
                            [--temperature TEMPERATURE]
options:
    --device_target                   Device type. Default: "Ascend"
    --device_id                       ID of target device
    --metric_method                   The eval method including [BLEU]. Default: "BLEU"
    --do_train                        Enable train. Default: "false"
    --do_eval                         Enable evaluation. Default: "true"
    --eval_type                       The type of evaluation including [zero-shot, finetuned]. Default: "zero-shot"
    --epoch_num                       Epoch number. Default: 1
    --train_data_shuffle              Enable train data shuffle. Default: "true"
    --eval_data_shuffle               Enable eval data shuffle. Default: "false"
    --save_finetune_ckpt_path         Save the checkpoint path
    --load_pretrain_ckpt_path         Load the checkpoint file path
    --load_finetune_ckpt_path         Load the checkpoint file path
    --train_data_file_path            Data path, it is better to use absolute path
    --eval_data_file_path             Data path, it is better to use absolute path
    --tokenizer_file_path             pretrained vocab and merge file path
    --generate_length                 The generation length of translation sentence
    --top_k                           Parameter for Top-K sampling
    --top_p                           Parameter for Top-P sampling
    --temperature                     Parameter for generation, greater if generation more diverse

```

# 环境要求

## 平台

- 硬件（Ascend）
    - 使用Ascend处理器准备硬件环境。- 如需试用昇腾处理器，请发送[申请表](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/file/other/Ascend%20Model%20Zoo%E4%BD%93%E9%AA%8C%E8%B5%84%E6%BA%90%E7%94%B3%E8%AF%B7%E8%A1%A8.docx)至ascend@huawei.com，申请通过即可获得资源。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/zh-CN/master/index.html)

## 其他要求

```text
math
numpy
copy
collections
re
rouge 1.0.0
datasets >=0.4.0
json
tensorflow
```

# 性能

## 推理性能

### Language Modeling任务

下表展示了GPT-2 small、medium、large三种规模的模型在Language Modeling任务中的PPL得分情况。

| 模型 | dataset | device | eval_type | PPL | OpenAI |
| :--- | :------ | :------ | :------ | :------ | :------ |
| GPT-2 small | WikiText2  | Ascend  | zero-shot | 24.5 | 29.41 |
| GPT-2 medium | WikiText2  | Ascend  | zero-shot | 19.41 | 22.76 |
| GPT-2 large | WikiText2  | Ascend  | zero-shot | 17.08 | 19.93 |
| GPT-2 small | WikiText103  | Ascend  | zero-shot | 26.89 | 37.5 |
| GPT-2 medium | WikiText103  | Ascend  | zero-shot | 20.23 | 26.37 |
| GPT-2 large | WikiText103  | Ascend  | zero-shot | 17.48 | 22.05 |
| GPT-2 small | PTB  | Ascend  | finetune | 23.91 | 65.85 |
| GPT-2 medium | PTB  | Ascend  | finetune | 20.06 | 47.33 |
| GPT-2 large | PTB  | Ascend  | finetune | 18.84 | 40.31 |
| GPT-2 small | 1BW  | Ascend  | zero-shot | 63.13 | 75.2 |
| GPT-2 medium | 1BW  | Ascend  | zero-shot | 50.98 | 55.72 |
| GPT-2 large | 1BW  | Ascend  | finetune | 29.28 | 44.575 |

### Children's Book Test 任务

下表展示了GPT-2 small、medium、large三种规模的模型在Children's Book Test 任务中的Accuracy得分情况。

| 模型 | dataset | device | eval_type | ACC | OpenAI |
| :--- | :------ | :------ | :------ | :------ | :------ |
| GPT-2 small | CBT-CN valid  | Ascend  | zero-shot | 87.85 | 87.65 |
| GPT-2 medium | CBT-CN valid  | Ascend  | zero-shot | 92.1 | 92.35 |
| GPT-2 large | CBT-CN valid  | Ascend  | zero-shot | 93.7 | 93.45 |
| GPT-2 small | CBT-NE valid  | Ascend  | zero-shot | 85.1 | 83.4 |
| GPT-2 medium | CBT-NE valid  | Ascend  | zero-shot | 87.55 | 87.1 |
| GPT-2 large | CBT-NE valid  | Ascend  | zero-shot | 89.1 | 88 |

### LAMBADA 任务

下表展示了GPT-2 small、medium、large三种规模的模型在LAMBADA 任务中的Accuracy和PPL得分情况。

| 模型 | dataset | device | eval_type | ACC | OpenAI |
| :--- | :------ | :------ | :------ | :------ | :------ |
| GPT-2 small | Lambada-test  | Ascend  | zero-shot | 45.99 | 45.99 |
| GPT-2 medium | Lambada-test  | Ascend  | zero-shot | 58.59 | 55.48 |
| GPT-2 large | Lambada-test  | Ascend  | zero-shot | 62.74 | 60.12 |

| 模型 | dataset | device | eval_type | PPL | OpenAI |
| :--- | :------ | :------ | :------ | :------ | :------ |
| GPT-2 small | Lambada-test  | Ascend  | zero-shot | 22.95 | 35.13 |
| GPT-2 medium | Lambada-test  | Ascend  | zero-shot | 10.69 | 15.6 |
| GPT-2 large | Lambada-test  | Ascend  | zero-shot | 8.64 | 10.87 |

### Reading Comprehension 任务

下表展示了GPT-2 small、medium、large三种规模的模型在Reading Comprehension任务中的F1得分情况。

| 模型 | dataset | device | eval_type | F1 | OpenAI |
| :--- | :------ | :------ | :------ | :------ | :------ |
| GPT-2 small | CoQA  | Ascend  | zero-shot | 25.94 | 25~26 |
| GPT-2 medium | CoQA  | Ascend  | zero-shot | 43.69 | 42~43 |
| GPT-2 large | CoQA  | Ascend  | zero-shot | 49.39 | 49~51 |

### Summarization 任务

下表展示了GPT-2 small、medium、large三种规模的模型在Summarization任务中的ROUGE得分情况。

| 模型 | dataset | device | eval_type | ROUGE | OpenAI |
| :--- | :------ | :------ | :------ | :------ | :------ |
| GPT-2 small | CNN_Dailymail(TL;DR)  | Ascend  | finetune | 21.4 | 16.8~17 |
| GPT-2 medium | CNN_Dailymail(TL;DR)  | Ascend  | finetune | 25.94 | 20.6~20.9 |
| GPT-2 large | CNN_Dailymail(TL;DR)  | Ascend  | finetune | 26.73 | 21.5~21.6 |

| 模型 | dataset | device | eval_type | ROUGE | OpenAI |
| :--- | :------ | :------ | :------ | :------ | :------ |
| GPT-2 small | CNN_Dailymail(no hint)  | Ascend  | zero-shot | 12.08 | 15.03(xlarge) |
| GPT-2 medium | CNN_Dailymail(no hint)  | Ascend  | zero-shot | 12.16 | 15.03(xlarge) |
| GPT-2 large | CNN_Dailymail(no hint)  | Ascend  | zero-shot | 12.29 | 15.03(xlarge) |

### Translation 任务

下表展示了GPT-2 small、medium、large三种规模的模型在Translation任务中的BLEU得分情况。

| 模型 | dataset | device | eval_type | BLEU | OpenAI |
| :--- | :------ | :------ | :------ | :------ | :------ |
| GPT-2 small | WMT-14 Fr-En  | Ascend  | zero-shot | 4.49 | 0.7~0.8 |
| GPT-2 medium | WMT-14 Fr-En  | Ascend  | zero-shot | 7.09 | 2.0~3.0 |
| GPT-2 large | WMT-14 Fr-En  | Ascend  | zero-shot | 7.97 | 6.5~7.0 |
| GPT-2 small | WMT-14 En-Fr  | Ascend  | zero-shot | 2.81 | 5(xlarge) |
| GPT-2 medium | WMT-14 En-Fr  | Ascend  | zero-shot | 3.2 | 5(xlarge) |
| GPT-2 large | WMT-14 En-Fr  | Ascend  | zero-shot | 3.06 | 5(xlarge) |

# 其他

该模型已在Ascend环境下环境下得到验证。

# ModelZoo主页  

 [链接](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)