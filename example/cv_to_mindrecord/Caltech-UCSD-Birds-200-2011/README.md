# Guideline to Transfer Caltech-UCSD Birds-200-2011 Dataset to MindRecord

<!-- TOC -->

- [What does the example do](#what-does-the-example-do)
- [How to use the example to generate MindRecord](#how-to-use-the-example-to-generate-mindrecord)
    - [Download Caltech-UCSD Birds-200-2011 dataset and unzip](#download-caltech-ucsd-birds-200-2011-dataset-and-unzip)
    - [Generate MindRecord](#generate-mindrecord)
    - [Create MindDataset By MindRecord](#create-minddataset-by-mindrecord)


<!-- /TOC -->

## What does the example do

This example is used to read data from Caltech-UCSD Birds-200-2011 dataset and generate mindrecord. It just transfers the Caltech-UCSD Birds-200-2011 dataset to mindrecord without any data preprocessing. You can modify the example or follow the example to implement your own example.

1.  run.sh: generate MindRecord entry script.
    - gen_mindrecord.py : read the Caltech-UCSD Birds-200-2011 data and transfer it to mindrecord.
2.  run_read.py: create MindDataset by MindRecord entry script.
    - create_dataset.py: use MindDataset to read MindRecord to generate dataset.

## How to use the example to generate MindRecord

Download Caltech-UCSD Birds-200-2011 dataset, transfer it to mindrecord, use MindDataset to read mindrecord.

### Download Caltech-UCSD Birds-200-2011 dataset and unzip

1. Download the training data zip.
    > [Caltech-UCSD Birds-200-2011 dataset download address](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)  
    > **1) -> Download -> All Images and Annotations**  
    > **2) -> Download -> Segmentations**  

2. Unzip the training data to dir example/nlp_to_mindrecord/Caltech-UCSD-Birds-200-2011/data.
    ```
    tar -zxvf CUB_200_2011.tgz -C {your-mindspore}/example/cv_to_mindrecord/Caltech-UCSD-Birds-200-2011/data/
    tar -zxvf segmentations.tgz -C {your-mindspore}/example/cv_to_mindrecord/Caltech-UCSD-Birds-200-2011/data/
    ```
    - The unzip should like this:
    ```
    $ ls {your-mindspore}/example/cv_to_mindrecord/Caltech-UCSD-Birds-200-2011/data/
    attributes.txt  CUB_200_2011  README.md  segmentations
    ```

### Generate MindRecord

1. Run the run.sh script.
    ```bash
    bash run.sh
    ```

2. Output like this:
    ```
    ...
    >> begin generate mindrecord
    >> sample id: 1, filename: data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg, bbox: [60.0, 27.0, 325.0, 304.0], label: 1, seg_filename: data/segmentations/001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.png, class: 001.Black_footed_Albatross
    [INFO] MD(11253,python):2020-05-20-16:21:42.462.686 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:106] OpenDataFiles] Open shard file successfully.
    [INFO] MD(11253,python):2020-05-20-16:21:43.147.496 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:680] WriteRawData] Write 256 records successfully.
    >> transformed 256 record...
    [INFO] MD(11253,python):2020-05-20-16:21:43.842.372 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:680] WriteRawData] Write 256 records successfully.
    >> transformed 512 record...
     [INFO] MD(11253,python):2020-05-20-16:21:44.748.585 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:680] WriteRawData] Write 256 records successfully.
    >> transformed 768 record...
    [INFO] MD(11253,python):2020-05-20-16:21:45.736.179 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:680] WriteRawData] Write 256 records successfully.
    >> transformed 1024 record...
    ...
    [INFO] MD(11253,python):2020-05-20-16:22:21.207.820 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:680] WriteRawData] Write 12 records successfully.
    >> transformed 11788 record...
    [INFO] MD(11253,python):2020-05-20-16:22:21.210.261 [mindspore/ccsrc/mindrecord/io/shard_writer.cc:227] Commit] Write metadata successfully.
    [INFO] MD(11253,python):2020-05-20-16:22:21.211.688 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:59] Build] Init header from mindrecord file for index successfully.
    [INFO] MD(11253,python):2020-05-20-16:22:21.236.799 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:600] DatabaseWriter] Init index db for shard: 0 successfully.
    [INFO] MD(11253,python):2020-05-20-16:22:21.964.034 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:549] ExecuteTransaction] Insert 11788 rows to index db.
    [INFO] MD(11253,python):2020-05-20-16:22:21.978.087 [mindspore/ccsrc/mindrecord/io/shard_index_generator.cc:620] DatabaseWriter] Generate index db for shard: 0 successfully.
    [INFO] ME(11253:139923799271232,MainProcess):2020-05-20-16:22:21.979.634 [mindspore/mindrecord/filewriter.py:313] The list of mindrecord files created are: ['output/CUB_200_2011.mindrecord'], and the list of index files are: ['output/CUB_200_2011.mindrecord.db']
    ```

3. Generate mindrecord files
    ```
    $ ls output/
    CUB_200_2011.mindrecord  CUB_200_2011.mindrecord.db  README.md
    ```

### Create MindDataset By MindRecord

1. Run the run_read.sh script.
    ```bash
    bash run_read.sh
    ```

2. Output like this:
    ```
    [INFO] MD(12469,python):2020-05-20-16:26:38.308.797 [mindspore/ccsrc/dataset/util/task.cc:31] operator()] Op launched, OperatorId:0 Thread ID 139702598620928 Started.
    [INFO] MD(12469,python):2020-05-20-16:26:38.322.433 [mindspore/ccsrc/mindrecord/io/shard_reader.cc:343] ReadAllRowsInShard] Get 11788 records from shard 0 index.
    [INFO] MD(12469,python):2020-05-20-16:26:38.386.904 [mindspore/ccsrc/mindrecord/io/shard_reader.cc:1058] CreateTasks] Total rows is 11788
    [INFO] MD(12469,python):2020-05-20-16:26:38.387.068 [mindspore/ccsrc/dataset/util/task.cc:31] operator()] Parallel Op Worker Thread ID 139702590228224 Started.
    [INFO] MD(12469,python):2020-05-20-16:26:38.387.272 [mindspore/ccsrc/dataset/util/task.cc:31] operator()] Parallel Op Worker Thread ID 139702581044992 Started.
    [INFO] MD(12469,python):2020-05-20-16:26:38.387.465 [mindspore/ccsrc/dataset/util/task.cc:31] operator()] Parallel Op Worker Thread ID 139702572652288 Started.
    [INFO] MD(12469,python):2020-05-20-16:26:38.387.617 [mindspore/ccsrc/dataset/util/task.cc:31] operator()] Parallel Op Worker Thread ID 139702564259584 Started.
    example 0: {'image': array([255, 216, 255, ...,  47, 255, 217], dtype=uint8), 'bbox': array([ 70., 120., 168., 150.], dtype=float32), 'label': array(199, dtype=int32), 'image_filename': array([ 87, 105, 110, 116, 101, 114,  95,  87, 114, 101, 110,  95,  48,
        49,  49,  54,  95,  49,  56,  57,  56,  51,  52,  46, 106, 112,
       103], dtype=uint8), 'segmentation_mask': array([137,  80,  78, ...,  66,  96, 130], dtype=uint8), 'label_name': array([ 49,  57,  57,  46,  87, 105, 110, 116, 101, 114,  95,  87, 114,
       101, 110], dtype=uint8)}
    example 1: {'image': array([255, 216, 255, ...,   3, 255, 217], dtype=uint8), 'bbox': array([ 51.,  51., 235., 322.], dtype=float32), 'label': array(170, dtype=int32), 'image_filename': array([ 77, 111, 117, 114, 110, 105, 110, 103,  95,  87,  97, 114,  98,
       108, 101, 114,  95,  48,  48,  55,  52,  95,  55,  57,  53,  51,
        54,  55,  46, 106, 112, 103], dtype=uint8), 'segmentation_mask': array([137,  80,  78, ...,  66,  96, 130], dtype=uint8), 'label_name': array([ 49,  55,  48,  46,  77, 111, 117, 114, 110, 105, 110, 103,  95,
        87,  97, 114,  98, 108, 101, 114], dtype=uint8)}
    example 2: {'image': array([255, 216, 255, ...,  35, 255, 217], dtype=uint8), 'bbox': array([ 57.,  56., 285., 248.], dtype=float32), 'label': array(148, dtype=int32), 'image_filename': array([ 71, 114, 101, 101, 110,  95,  84,  97, 105, 108, 101, 100,  95,
        84, 111, 119, 104, 101, 101,  95,  48,  48,  53,  52,  95,  49,
        53,  52,  57,  51,  56,  46, 106, 112, 103], dtype=uint8), 'segmentation_mask': array([137,  80,  78, ...,  66,  96, 130], dtype=uint8), 'label_name': array([ 49,  52,  56,  46,  71, 114, 101, 101, 110,  95, 116,  97, 105,
       108, 101, 100,  95,  84, 111, 119, 104, 101, 101], dtype=uint8)}
    example 3: {'image': array([255, 216, 255, ...,  85, 255, 217], dtype=uint8), 'bbox': array([ 95.,  61., 333., 323.], dtype=float32), 'label': array(176, dtype=int32), 'image_filename': array([ 80, 114,  97, 105, 114, 105, 101,  95,  87,  97, 114,  98, 108,
       101, 114,  95,  48,  49,  48,  53,  95,  49,  55,  50,  57,  56,
        50,  46, 106, 112, 103], dtype=uint8), 'segmentation_mask': array([137,  80,  78, ...,  66,  96, 130], dtype=uint8), 'label_name': array([ 49,  55,  54,  46,  80, 114,  97, 105, 114, 105, 101,  95,  87,
        97, 114,  98, 108, 101, 114], dtype=uint8)}
    ...
    example 11786: {'image': array([255, 216, 255, ..., 199, 255, 217], dtype=uint8), 'bbox': array([180.,  61., 153., 162.], dtype=float32), 'label': array(75, dtype=int32), 'image_filename': array([ 71, 114, 101, 101, 110,  95,  74,  97, 121,  95,  48,  48,  55,
        49,  95,  54,  53,  55,  57,  57,  46, 106, 112, 103], dtype=uint8), 'segmentation_mask': array([137,  80,  78, ...,  66,  96, 130], dtype=uint8), 'label_name': array([ 48,  55,  53,  46,  71, 114, 101, 101, 110,  95,  74,  97, 121],
      dtype=uint8)}
    example 11787: {'image': array([255, 216, 255, ..., 127, 255, 217], dtype=uint8), 'bbox': array([ 49.,  33., 276., 216.], dtype=float32), 'label': array(27, dtype=int32), 'image_filename': array([ 83, 104, 105, 110, 121,  95,  67, 111, 119,  98, 105, 114, 100,
        95,  48,  48,  51,  49,  95,  55,  57,  54,  56,  53,  49,  46,
       106, 112, 103], dtype=uint8), 'segmentation_mask': array([137,  80,  78, ...,  66,  96, 130], dtype=uint8), 'label_name': array([ 48,  50,  55,  46,  83, 104, 105, 110, 121,  95,  67, 111, 119,
        98, 105, 114, 100], dtype=uint8)}
    >> total rows: 11788
    [INFO] MD(12469,python):2020-05-20-16:26:49.582.298 [mindspore/ccsrc/dataset/util/task.cc:128] Join] Watchdog Thread ID 139702607013632 Stopped.
    ```
    - bbox : coordinate value of the bounding box in the picture.
    - image: the image bytes which is from like "data/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg".
    - image_filename: the image name which is like "Black_Footed_Albatross_0001_796111.jpg"
    - label : the picture label which is in [1, 200].
    - lable_name : object which is like "016.Painted_Bunting" corresponding to label.
    - segmentation_mask : the image bytes which is from like "data/segmentations/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.png".
