# Guideline to Efficiently Generating MindRecord

<!-- TOC -->

- [What does the example do](#what-does-the-example-do)
- [Example test for ImageNet](#example-test-for-imagenet)
- [How to use the example for other dataset](#how-to-use-the-example-for-other-dataset)
    - [Create work space](#create-work-space)
    - [Implement data generator](#implement-data-generator)
    - [Run data generator](#run-data-generator)


<!-- /TOC -->

## What does the example do

This example provides an efficient way to generate MindRecord. Users only need to define the parallel granularity of training data reading and the data reading function of a single task. That is, they can efficiently convert the user's training data into MindRecord.

1.  run_template.sh: entry script, users need to modify parameters according to their own training data.
2.  writer.py: main script, called by run_template.sh, it mainly reads user training data in parallel and generates MindRecord.
3.  template/mr_api.py: uers define their own parallel granularity of training data reading and single task reading function through the template.

## Example test for ImageNet

1. Download and prepare the ImageNet dataset as required.

    > [ImageNet dataset download address](http://image-net.org/download)

    Store the downloaded ImageNet dataset in a folder. The folder contains all images and a mapping file that records labels of the images.

    In the mapping file, there are three columns, which are separated by spaces. They indicate image classes, label IDs, and label names. The following is an example of the mapping file:
    ```
    n02119789 1 pen
    n02100735 2 notbook
    n02110185 3 mouse
    n02096294 4 orange
    ```
2. Edit run_imagenet.sh and modify the parameters
3. Run the bash script  
    ```bash  
    bash run_imagenet.sh
    ```  
4. Performance result

    |  Training Data |  General API | Current Example |  Env  |
    | ---- | ---- | ---- | ---- |
    |ImageNet(140G)|  2h40m |  50m  |  CPU: Intel Xeon Gold 6130 x 64, Memory: 256G, Storage: HDD |

## How to use the example for other dataset
### Create work space

Assume the dataset name is 'xyz'
* Create work space from template
    ```shell
    cd ${your_mindspore_home}/example/convert_to_mindrecord
    cp -r template xyz
    ```

### Implement data generator

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
- For ImageNet, one directory can be a task.
- For TFRecord with multiple files, each file can be a task.
- For TFRecord with 1 file only, it could also be split into N tasks. Task_id=K means: data row is picked only if (count % N == K) 

### Run data generator

* run python script 
    ```shell
    cd ${your_mindspore_home}/example/convert_to_mindrecord
    python writer.py --mindrecord_script xyz [...]
    ```
    > You can put this command in script **run_xyz.sh** for easy execution

