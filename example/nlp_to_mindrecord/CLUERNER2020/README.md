# Guideline to Convert Training Data CLUERNER2020 to MindRecord For Bert Fine Tuning

<!-- TOC -->

- [What does the example do](#what-does-the-example-do)
- [How to use the example to process CLUERNER2020](#how-to-use-the-example-to-process-cluerner2020)
    - [Download CLUERNER2020 and unzip](#download-cluerner2020-and-unzip)
    - [Generate MindRecord](#generate-mindrecord)
    - [Create MindDataset By MindRecord](#create-minddataset-by-mindrecord)


<!-- /TOC -->

## What does the example do

This example is based on [CLUERNER2020](https://www.cluebenchmarks.com/introduce.html) training data, generating MindRecord file, and finally used for Bert Fine Tuning progress.

1.  run.sh: generate MindRecord entry script
    - data_processor_seq.py: the script from [CLUEbenchmark/CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020/tree/master/tf_version), we just change the part of the generated tfrecord to MindRecord.
    - label2id.json: the file from [CLUEbenchmark/CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020/tree/master/tf_version).
    - tokenization.py: the script from [CLUEbenchmark/CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020/tree/master/tf_version).
    - vocab.txt: the file from [CLUEbenchmark/CLUENER2020](https://github.com/CLUEbenchmark/CLUENER2020/tree/master/tf_version).
2.  run_read.py: create MindDataset by MindRecord entry script.
    - create_dataset.py: use MindDataset to read MindRecord to generate dataset.
3. data: the output directory for MindRecord.
4. cluener_public: the CLUENER2020 training data.

## How to use the example to process CLUERNER2020

Download CLUERNER2020, convert it to MindRecord, use MindDataset to read MindRecord.

### Download CLUERNER2020 and unzip

1. Download the training data zip.
    > [CLUERNER2020 dataset download address](https://www.cluebenchmarks.com/introduce.html) **-> 任务介绍 -> CLUENER 细粒度命名实体识别 -> cluener下载链接**

2. Unzip the training data to dir example/nlp_to_mindrecord/CLUERNER2020/cluener_public.
    ```
    unzip -d {your-mindspore}/example/nlp_to_mindrecord/CLUERNER2020/cluener_public cluener_public.zip
    ```

### Generate MindRecord

1. Run the run.sh script.
    ```bash
    bash run.sh
    ```

2. Output like this:
    ```
    ...
    [INFO] ME(17603:139620983514944,MainProcess):2020-04-28-16:56:12.498.235 [mindspore/mindrecord/filewriter.py:313] The list of mindrecord files created are: ['data/train.mindrecord'], and the list of index files are: ['data/train.mindrecord.db']
    ...
    [INFO] ME(17603,python):2020-04-28-16:56:13.400.175 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:667] WriteRawData] Write 1 records successfully.
    [INFO] ME(17603,python):2020-04-28-16:56:13.400.863 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:667] WriteRawData] Write 1 records successfully.
    [INFO] ME(17603,python):2020-04-28-16:56:13.401.534 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:667] WriteRawData] Write 1 records successfully.
    [INFO] ME(17603,python):2020-04-28-16:56:13.402.179 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:667] WriteRawData] Write 1 records successfully.
    [INFO] ME(17603,python):2020-04-28-16:56:13.402.702 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:667] WriteRawData] Write 1 records successfully.
    ...
    [INFO] ME(17603:139620983514944,MainProcess):2020-04-28-16:56:13.431.208 [mindspore/mindrecord/filewriter.py:313] The list of mindrecord files created are: ['data/dev.mindrecord'], and the list of index files are: ['data/dev.mindrecord.db']
    ```

### Create MindDataset By MindRecord

1. Run the run_read.sh script.
    ```bash
    bash run_read.sh
    ```

2. Output like this:
    ```
    ...
    example 1340: input_ids: [ 101 3173 1290 4852 7676 3949  122 3299  123  126 3189 4510 8020 6381 5442 7357 2590 3636 8021 7676 3949 4294 1166 6121 3124 1277 6121 3124 7270 2135 3295 5789 3326 123  126 3189 1355 6134 1093 1325 3173 2399 6590 6791 8024  102    0    0    0    0    0    0    0    0    0    0   0    0    0    0    0    0    0    0]
    example 1340: input_mask: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1  1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    example 1340: segment_ids: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    example 1340: label_ids: [ 0 18 19 20  2  4  0  0  0  0  0  0  0 34 36 26 27 28  0 34 35 35 35 35 35 35 35 35 35 36 26 27 28  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    example 1341: input_ids: [ 101 1728  711 4293 3868 1168 2190 2150 3791  934 3633 3428 4638 6237 7025 8024 3297 1400 5310 3362 6206 5023 5401 1744 3297 7770 3791 7368  976 1139 1104 2137  511 102    0    0    0    0    0    0    0    0   0    0    0    0    0    0    0    0    0    0    0    0    0    0   0    0    0    0    0    0    0    0]
    example 1341: input_mask: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    example 1341: segment_ids: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
   example 1341: label_ids: [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 18 19 19 19 19 20  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
    ...
    ```
