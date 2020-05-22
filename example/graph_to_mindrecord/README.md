# Guideline to Efficiently Generating MindRecord

<!-- TOC -->

- [What does the example do](#what-does-the-example-do)
- [Example test for Cora](#example-test-for-cora)
- [How to use the example for other dataset](#how-to-use-the-example-for-other-dataset)
    - [Create work space](#create-work-space)
    - [Implement data generator](#implement-data-generator)
    - [Run data generator](#run-data-generator)


<!-- /TOC -->

## What does the example do

This example provides an efficient way to generate MindRecord. Users only need to define the parallel granularity of training data reading and the data reading function of a single task. That is, they can efficiently convert the user's training data into MindRecord.

1.  write_cora.sh: entry script, users need to modify parameters according to their own training data.
2.  writer.py: main script, called by write_cora.sh, it mainly reads user training data in parallel and generates MindRecord.
3.  cora/mr_api.py: uers define their own parallel granularity of training data reading and single task reading function through the cora.

## Example test for Cora

1. Download and prepare the Cora dataset as required.

    > [Cora dataset download address](https://github.com/jzaldi/datasets/tree/master/cora)


2. Edit write_cora.sh and modify the parameters
    ```
    --mindrecord_file: output MindRecord file.
    --mindrecord_partitions: the partitions for MindRecord.
    ```

3. Run the bash script
    ```bash  
    bash write_cora.sh
    ```  

## How to use the example for other dataset

### Create work space

Assume the dataset name is 'xyz'
* Create work space from cora
    ```shell
    cd ${your_mindspore_home}/example/graph_to_mindrecord
    cp -r cora xyz
    ```

### Implement data generator

Edit dictionary data generator.
* Edit file 
    ```shell
    cd ${your_mindspore_home}/example/graph_to_mindrecord
    vi xyz/mr_api.py
    ```

Two API, 'mindrecord_task_number' and 'mindrecord_dict_data', must be implemented.
- 'mindrecord_task_number()' returns number of tasks. Return 1 if data row is generated serially. Return N if generator can be split into N parallel-run tasks.
- 'mindrecord_dict_data(task_id)' yields dictionary data row by row. 'task_id' is 0..N-1, if N is return value of mindrecord_task_number()

### Run data generator

* run python script 
    ```shell
    cd ${your_mindspore_home}/example/graph_to_mindrecord
    python writer.py --mindrecord_script xyz [...]
    ```
    > You can put this command in script **write_xyz.sh** for easy execution

