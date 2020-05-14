# Guideline to Preprocess Large Movie Review Dataset - aclImdb to MindRecord

<!-- TOC -->

- [What does the example do](#what-does-the-example-do)
- [How to use the example to generate MindRecord](#how-to-use-the-example-to-generate-mindrecord)
    - [Download aclImdb dataset and unzip](#download-aclimdb-dataset-and-unzip)
    - [Generate MindRecord](#generate-mindrecord)
    - [Create MindDataset By MindRecord](#create-minddataset-by-mindrecord)


<!-- /TOC -->

## What does the example do

This example is used to read data from aclImdb dataset, preprocess it and generate mindrecord. The preprocessing process mainly uses vocab file to convert the training set text into dictionary sequence, which can be further used in the subsequent training process.

1.  run.sh: generate MindRecord entry script.
    - gen_mindrecord.py : read the aclImdb data, preprocess it and transfer it to mindrecord.
2.  run_read.py: create MindDataset by MindRecord entry script.
    - create_dataset.py: use MindDataset to read MindRecord to generate dataset.

## How to use the example to generate MindRecord

Download aclImdb dataset, transfer it to mindrecord, use MindDataset to read mindrecord.

### Download aclImdb dataset and unzip

1. Download the training data zip.
    > [aclImdb dataset download address](http://ai.stanford.edu/~amaas/data/sentiment/) **-> Large Movie Review Dataset v1.0**

2. Unzip the training data to dir example/nlp_to_mindrecord/aclImdb_preprocess/data.
    ```
    tar -zxvf aclImdb_v1.tar.gz -C {your-mindspore}/example/nlp_to_mindrecord/aclImdb_preprocess/data/
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
    >> transformed 256 record...
    >> transformed 512 record...
    >> transformed 768 record...
    >> transformed 1024 record...
    ...
    >> transformed 25000 record...
    [INFO] MD(6553,python):2020-05-14-16:10:44.947.617 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:227] Commit] Write metadata successfully.
    [INFO] MD(6553,python):2020-05-14-16:10:44.948.193 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:59] Build] Init header from mindrecord file for index successfully.
    [INFO] MD(6553,python):2020-05-14-16:10:44.974.544 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:600] DatabaseWriter] Init index db for shard: 0 successfully.
    [INFO] MD(6553,python):2020-05-14-16:10:46.110.119 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:549] ExecuteTransaction] Insert 25000 rows to index db.
    [INFO] MD(6553,python):2020-05-14-16:10:46.128.212 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:620] DatabaseWriter] Generate index db for shard: 0 successfully.
    [INFO] ME(6553:139716072798016,MainProcess):2020-05-14-16:10:46.130.596 [mindspore/mindrecord/filewriter.py:313] The list of mindrecord files created are: ['output/aclImdb_train.mindrecord'], and the list of index files are: ['output/aclImdb_train.mindrecord.db']
    >> begin generate mindrecord by test data
    >> transformed 256 record...
    >> transformed 512 record...
    >> transformed 768 record...
    >> transformed 1024 record...
    ...
    [INFO] MD(6553,python):2020-05-14-16:10:55.047.633 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:600] DatabaseWriter] Init index db for shard: 0 successfully.
    [INFO] MD(6553,python):2020-05-14-16:10:56.092.477 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:549] ExecuteTransaction] Insert 25000 rows to index db.
    [INFO] MD(6553,python):2020-05-14-16:10:56.107.799 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:620] DatabaseWriter] Generate index db for shard: 0 successfully.
    [INFO] ME(6553:139716072798016,MainProcess):2020-05-14-16:10:56.111.193 [mindspore/mindrecord/filewriter.py:313] The list of mindrecord files created are: ['output/aclImdb_test.mindrecord'], and the list of index files are: ['output/aclImdb_test.mindrecord.db']
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
    ```
    example 24992: {'input_ids': array([   -1,    -1,    65,     0,    89,     0,   367,     0,    -1,
          -1,    -1,    -1,   488,     0,     0,     0,   206,     0,
         816,     0,    -1,    -1,    16,     0,    -1,    -1, 11998,
           0,     0,     0,   852,     0,     1,     0,   111,     0,
          -1,    -1,    -1,    -1,   765,     0,     9,     0,    17,
           0,    35,     0,    72,     0,    -1,    -1,    -1,    -1,
          40,     0,   895,     0,    41,     0,     0,     0,  6952,
           0,   170,     0,    -1,    -1,    -1,    -1,     3,     0,
          28,     0,    -1,    -1,     0,     0,   111,     0,    58,
           0,   110,     0,   569,     0,    -1,    -1,    -1,    -1,
          -1,    -1,     0,     0, 24512,     0,     3,     0,     0,
           0], dtype=int32), 'id': array(8045, dtype=int32), 'input_mask': array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=int32), 'segment_ids': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32), 'score': array(1, dtype=int32), 'label': array(1, dtype=int32)}
    example 24993: {'input_ids': array([  -1,   -1,   11,    0, 7400,    0,  189,    0,    4,    0, 1247,
          0,    9,    0,   17,    0,   29,    0,    0,    0,   -1,   -1,
         -1,   -1,   -1,   -1,    1,    0,   -1,   -1,  218,    0,  131,
          0,   10,    0,   -1,   -1,   52,    0,   72,    0,  488,    0,
          6,    0,   -1,   -1,   -1,   -1,   -1,   -1, 1749,    0,    0,
          0,   -1,   -1,   42,    0,   21,    0,   65,    0, 6895,    0,
         -1,   -1,   -1,   -1,   -1,   -1,   11,    0,   52,    0,   72,
          0, 1498,    0,   10,    0,   21,    0,   65,    0,   19,    0,
         -1,   -1,   -1,   -1,   36,    0,  130,    0,   88,    0,  210,
          0], dtype=int32), 'id': array(9903, dtype=int32), 'input_mask': array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
       1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=int32), 'segment_ids': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32), 'score': array(7, dtype=int32), 'label': array(0, dtype=int32)}
    ```
    - id : the id "3219" is from review docs like **3219**_10.txt.
    - label : indicates whether the review is positive or negative, positive: 0, negative: 1.
    - score : the score "10" is from review docs like 3219_**10**.txt.
    - input_ids : the input_ids are from the review dos's content which mapped by imdb.vocab file.
    - input_mask : the input_mask are from the review dos's content which mapped by imdb.vocab file.
    - segment_ids : the segment_ids are from the review dos's content which mapped by imdb.vocab file.
