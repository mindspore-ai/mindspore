# Guideline to Transfer Large Movie Review Dataset - aclImdb to MindRecord

<!-- TOC -->

- [What does the example do](#what-does-the-example-do)
- [How to use the example to generate MindRecord](#how-to-use-the-example-to-generate-mindrecord)
    - [Download aclImdb dataset and unzip](#download-aclimdb-dataset-and-unzip)
    - [Generate MindRecord](#generate-mindrecord)
    - [Create MindDataset By MindRecord](#create-minddataset-by-mindrecord)

<!-- /TOC -->

## What does the example do

This example is used to read data from aclImdb dataset and generate mindrecord. It just transfers the aclImdb dataset to mindrecord without any data preprocessing. You can modify the example or refer to the example to implement your own example.

1. run.sh: generate MindRecord entry script.
    - gen_mindrecord.py : read the aclImdb data and transfer it to mindrecord.
2. run_read.py: create MindDataset by MindRecord entry script.
    - create_dataset.py: use MindDataset to read MindRecord to generate dataset.

## How to use the example to generate MindRecord

Download aclImdb dataset, transfer it to mindrecord, use MindDataset to read mindrecord.

### Download aclImdb dataset and unzip

1. Download the training data zip.
    > [aclImdb dataset download address](http://ai.stanford.edu/~amaas/data/sentiment/) **-> Large Movie Review Dataset v1.0**

2. Unzip the training data to dir example/nlp_to_mindrecord/aclImdb/data.

    ```
    tar -zxvf aclImdb_v1.tar.gz -C {your-mindspore}/example/nlp_to_mindrecord/aclImdb/data/
    ```

### Generate MindRecord

1. Run the run.sh script.

    ```bash
    bash run.sh
    ```

2. Output like this:

    ```
    ...
    >> begin generate mindrecord by train data
    ...
    [INFO] ME(20928,python):2020-05-07-23:02:40.066.546 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:667] WriteRawData] Write 256 records successfully.
    >> transformed 24320 record...
    [INFO] ME(20928,python):2020-05-07-23:02:40.078.344 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:667] WriteRawData] Write 256 records successfully.
    >> transformed 24576 record...
    [INFO] ME(20928,python):2020-05-07-23:02:40.090.237 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:667] WriteRawData] Write 256 records successfully.
    >> transformed 24832 record...
    [INFO] ME(20928,python):2020-05-07-23:02:40.098.785 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:667] WriteRawData] Write 168 records successfully.
    >> transformed 25000 record...
    [INFO] ME(20928,python):2020-05-07-23:02:40.098.957 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:214] Commit] Write metadata successfully.
    [INFO] ME(20928,python):2020-05-07-23:02:40.099.302 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:45] Build] Init header from mindrecord file for index successfully.
    [INFO] ME(20928,python):2020-05-07-23:02:40.122.271 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:586] DatabaseWriter] Init index db for shard: 0 successfully.
    [INFO] ME(20928,python):2020-05-07-23:02:40.932.360 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:535] ExecuteTransaction] Insert 24596 rows to index db.
    [INFO] ME(20928,python):2020-05-07-23:02:40.953.177 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:535] ExecuteTransa ction] Insert 404 rows to index db.
    [INFO] ME(20928,python):2020-05-07-23:02:40.963.400 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:606] DatabaseWriter] Generate index db for shard: 0 successfully.
    [INFO] ME(20928:139630558652224,MainProcess):2020-05-07-23:02:40.964.973 [mindspore/mindrecord/filewriter.py:313] The list of mindrecord files created are: ['output/aclImdb_train.mindrecord'], and the list of index files are: ['output/aclImdb_train.mindrecord.db']
    >> begin generate mindrecord by test data
    ...
    >> transformed 24576 record...
    [INFO] ME(20928,python):2020-05-07-23:02:42.120.007 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:667] WriteRawData] Write 256 records successfully.
    >> transformed 24832 record...
    [INFO] ME(20928,python):2020-05-07-23:02:42.128.862 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:667] WriteRawData] Write 168 records successfully.
    >> transformed 25000 record...
    [INFO] ME(20928,python):2020-05-07-23:02:42.129.024 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:214] Commit] Write metadata successfully.
    [INFO] ME(20928,python):2020-05-07-23:02:42.129.362 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:45] Build] Init header from mindrecord file for index successfully.
    [INFO] ME(20928,python):2020-05-07-23:02:42.151.237 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:586] DatabaseWriter] Init index db for shard: 0 successfully.
    [INFO] ME(20928,python):2020-05-07-23:02:42.935.496 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:535] ExecuteTransaction] Insert 25000 rows to index db.
    [INFO] ME(20928,python):2020-05-07-23:02:42.949.319 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:606] DatabaseWriter] Generate index db for shard: 0 successfully.
    [INFO] ME(20928:139630558652224,MainProcess):2020-05-07-23:02:42.951.794 [mindspore/mindrecord/filewriter.py:313] The list of mindrecord files created are: ['output/aclImdb_test.mindrecord'], and the list of index files are: ['output/aclImdb_test.mindrecord.db']
    ```

3. Generate mindrecord files

    ```
    $ ls output/
    aclImdb_test.mindrecord  aclImdb_test.mindrecord.db  aclImdb_train.mindrecord  aclImdb_train.mindrecord.db  README.md
    ```

### Create MindDataset By MindRecord

1. Run the run_read.sh script.

    ```bash
    bash run_read.sh
    ```

2. Output like this:
    > Caution: field "review" which is string type output is displayed in type uint8.

    ```
    ...
    example 2056: {'label': array(1, dtype=int32), 'score': array(4, dtype=int32), 'id': array(5871, dtype=int32), 'review': array([ 70, 111, 114, ..., 111, 110,  46], dtype=uint8)}
    example 2057: {'label': array(1, dtype=int32), 'score': array(1, dtype=int32), 'id': array(6092, dtype=int32), 'review': array([ 83, 111, 109, ..., 115, 101,  46], dtype=uint8)}
    example 2058: {'label': array(1, dtype=int32), 'score': array(4, dtype=int32), 'id': array(1357, dtype=int32), 'review': array([ 42, 109,  97, ...,  58,  32,  67], dtype=uint8)}
    ...
    ```

   - id : the id "3219" is from review docs like **3219**_10.txt.
   - label : indicates whether the review is positive or negative, positive: 0, negative: 1.
   - score : the score "10" is from review docs like 3219_**10**.txt.
   - review : the review is from the review dos's content.
