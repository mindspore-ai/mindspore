# MindRecord generating guidelines

<!-- TOC -->

- [MindRecord generating guidelines](#mindrecord-generating-guidelines)
    - [Create work space](#create-work-space)
    - [Implement data generator](#implement-data-generator)
    - [Run data generator](#run-data-generator)

<!-- /TOC -->

## Create work space

Assume the dataset name is 'xyz'
* Create work space from template
    ```shell
    cd ${your_mindspore_home}/example/convert_to_mindrecord
    cp -r template xyz
    ```

## Implement data generator 

Edit dictionary data generator  
* Edit file 
    ```shell
    cd ${your_mindspore_home}/example/convert_to_mindrecord
    vi xyz/mr_api.py
    ```

 Two API, 'mindrecord_task_number' and 'mindrecord_dict_data', must be implemented
- 'mindrecord_task_number()' returns number of tasks. Return 1 if data row is generated serially. Return N if generator can be split into N parallel-run tasks.
- 'mindrecord_dict_data(task_id)' yields dictionary data row by row. 'task_id' is 0..N-1, if N is return value of mindrecord_task_number()


Tricky for parallel run
- For imagenet, one directory can be a task.
- For TFRecord with multiple files, each file can be a task.
- For TFRecord with 1 file only, it could also be split into N tasks. Task_id=K means: data row is picked only if (count % N == K) 


## Run data generator 
* run python script 
    ```shell
    cd ${your_mindspore_home}/example/convert_to_mindrecord
    python writer.py --mindrecord_script imagenet [...]
    ```
