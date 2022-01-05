# Federated Learning Image Classification Dataset Processing

In this example, the federated learning dataset `FEMNIST` in the `leaf` dataset is used. The dataset contains 62 different types of handwritten digits and letters (digits 0 to 9, lowercase letters, and uppercase letters). The image size is 28 x 28 pixels, the dataset contains handwritten digits and letters of 3500 users (a maximum of 3500 clients can be simulated to participate in federated learning). The total data volume is 805,263, the average data volume of each user is 226.83, and the variance of the data volume of all users is 88.94.

## Device-cloud federation learning image classification dataset processing

You can refer to [LEAF: A Benchmark for Federated Settings](https://github.com/TalwalkarLab/leaf).

each user is 226.83, and the variance of the data volume of all users is 88.94.

1. Ensure that the following environment requirements are met:

    ```sh
    numpy==1.16.4
    scipy                      # conda install scipy
    tensorflow==1.13.1         # pip install tensorflow
    Pillow                     # pip install Pillow
    matplotlib                 # pip install matplotlib
    jupyter                    # conda install jupyter notebook==5.7.8 tornado==4.5.3
    pandas                     # pip install pandas
    ```

2. Download the dataset generation script from GitHub:

    ```sh
    git clone https://github.com/TalwalkarLab/leaf.git
    ```

    After the project is downloaded, the directory structure is as follows:

    ```sh
    leaf-master/data/femnist
    ├── data  # Generated dataset
    ├── preprocess  # Code related to data preprocessing
    ├── preprocess.sh  # Shell script for generating the `femnist` dataset
    └── README.md  # Official dataset download guide
    ```

3. Take the `femnist` dataset as an example. Run the following command to go to the specified path:

    ```sh
    cd  leaf-master/data/femnist
    ```

4. Run the command `./preprocess.sh -s niid --sf 1.0 -k 0 -t sample` to generate a dataset containing data of 3500 users, and split data of each user into a training dataset and a test dataset at 9:1.

    For the meaning of the parameters in the command, please refer to the description in the `leaf/data/femnist/README.md` file.

    The directory structure is as follows:

    ```sh
    leaf-master/data/femnist/35_client_sf1_data/
    ├── all_data  # All datasets are mixed together. There are 35 JSON files in total, and each JSON file contains the data of 100 users.
    ├── test  # Test dataset obtained after splitting the data of each user into the training and test datasets at 9:1. There are 35 JSON files in total, and each JSON file contains the data of 100 users.
    ├── train  # Test dataset obtained after splitting the data of each user into the training and test datasets at 9:1. There are 35 JSON files in total, and each JSON file contains the data of 100 users.
    └── ...  # Other files are not involved and are not described here.
    ```

    Each JSON file consists of the following parts:

    - `users`: user list.
    - `num_samples`: sample quantity list of each user.
    - `user_data`: a dictionary object with the username as the key and their data as the value. For each user, the data is represented as an image list, and each image is represented as an integer list whose size is 784 (obtained by flattening the 28 x 28 image array).

    Before running the preprocess.sh script again, ensure that the rem_user_data, sampled_data, test, and train subfolders are deleted from the data directory.

5. Split 35 JSON files into 3500 JSON files (each JSON file represents a user).

    The sample code is as follows:

    ```python
    import os
    import json

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    def partition_json(root_path, new_root_path):
        """
        partition 35 json files to 3500 json file

        Each raw .json file is an object with 3 keys:
        1. 'users', a list of users
        2. 'num_samples', a list of the number of samples for each user
        3. 'user_data', an object with user names as keys and their respective data as values; for each user, data is represented as a list of images, with each image represented as a size-784 integer list (flattened from 28 by 28)

        Each new .json file is an object with 3 keys:
        1. 'user_name', the name of user
        2. 'num_samples', the number of samples for the user
        3. 'user_data', an dict object with 'x' as keys and their respective data as values; with 'y' as keys and their respective label as values;

        Args:
            root_path (str): raw root path of 35 json files
            new_root_path (str): new root path of 3500 json files
        """
        paths = os.listdir(root_path)
        count = 0
        file_num = 0
        for i in paths:
            file_num += 1
            file_path = os.path.join(root_path, i)
            print('======== process ' + str(file_num) + ' file: ' + str(file_path) + '======================')
            with open(file_path, 'r') as load_f:
                load_dict = json.load(load_f)
                users = load_dict['users']
                num_users = len(users)
                num_samples = load_dict['num_samples']
                for j in range(num_users):
                    count += 1
                    print('---processing user: ' + str(count) + '---')
                    cur_out = {'user_name': None, 'num_samples': None, 'user_data': {}}
                    cur_user_id = users[j]
                    cur_data_num = num_samples[j]
                    cur_user_path = os.path.join(new_root_path, cur_user_id + '.json')
                    cur_out['user_name'] = cur_user_id
                    cur_out['num_samples'] = cur_data_num
                    cur_out['user_data'].update(load_dict['user_data'][cur_user_id])
                    with open(cur_user_path, 'w') as f:
                        json.dump(cur_out, f)
        f = os.listdir(new_root_path)
        print(len(f), ' users have been processed!')
    # partition train json files
    partition_json("leaf-master/data/femnist/35_client_sf1_data/train", "leaf-master/data/femnist/3500_client_json/train")
    # partition test json files
    partition_json("leaf-master/data/femnist/35_client_sf1_data/test", "leaf-master/data/femnist/3500_client_json/test")
    ```

    In the preceding command, `root_path` is set to `leaf-master/data/femnist/35_client_sf1_data/{train,test}`, and `new_root_path` is set to a user-defined value to store the generated 3500 JSON files. The training and test folders need to be processed separately.

    A total of 3500 JSON files are generated. Each file contains the following parts:

    - `user_name`: username.
    - `num_samples`: number of user samples.
    - `user_data`: a dictionary object that uses 'x' as the key and user data as the value; uses 'y' as the key and the label corresponding to the user data as the value.

    If the following information is displayed, the script is successfully executed:

    ```sh
    ======== process 1 file: /leaf-master/data/femnist/35_client_sf1_data/train/all_data_16_niid_0_keep_0_train_9.json======================
    ---processing user: 1---
    ---processing user: 2---
    ---processing user: 3---
    ......
    ```

6. Convert the JSON file to an image file.

    For details, see the following code:

    ```python
    import os
    import json
    import numpy as np
    from PIL import Image

    name_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                 'V', 'W', 'X', 'Y', 'Z',
                 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                 'v', 'w', 'x', 'y', 'z'
                 ]

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    def json_2_numpy(img_size, file_path):
        """
        read json file to numpy
        Args:
            img_size (list): contain three elements: the height, width, channel of image
            file_path (str): root path of 3500 json files
        return:
            image_numpy (numpy)
            label_numpy (numpy)
        """
        # open json file
        with open(file_path, 'r') as load_f_train:
            load_dict = json.load(load_f_train)
            num_samples = load_dict['num_samples']
            x = load_dict['user_data']['x']
            y = load_dict['user_data']['y']
            size = (num_samples, img_size[0], img_size[1], img_size[2])
            image_numpy = np.array(x, dtype=np.float32).reshape(size)  # mindspore doesn't support float64 and int64
            label_numpy = np.array(y, dtype=np.int32)
        return image_numpy, label_numpy

    def json_2_img(json_path, save_path):
        """
        transform single json file to images

        Args:
            json_path (str): the path json file
            save_path (str): the root path to save images

        """
        data, label = json_2_numpy([28, 28, 1], json_path)
        for i in range(data.shape[0]):
            img = data[i] * 255  # PIL don't support the 0/1 image ,need convert to 0~255 image
            im = Image.fromarray(np.squeeze(img))
            im = im.convert('L')
            img_name = str(label[i]) + '_' + name_list[label[i]] + '_' + str(i) + '.png'
            path1 = os.path.join(save_path, str(label[i]))
            mkdir(path1)
            img_path = os.path.join(path1, img_name)
            im.save(img_path)
            print('-----', i, '-----')

    def all_json_2_img(root_path, save_root_path):
        """
        transform json files to images
        Args:
            json_path (str): the root path of 3500 json files
            save_path (str): the root path to save images
        """
        usage = ['train', 'test']
        for i in range(2):
            x = usage[i]
            files_path = os.path.join(root_path, x)
            files = os.listdir(files_path)

            for name in files:
                user_name = name.split('.')[0]
                json_path = os.path.join(files_path, name)
                save_path1 = os.path.join(save_root_path, user_name)
                mkdir(save_path1)
                save_path = os.path.join(save_path1, x)
                mkdir(save_path)
                print('=============================' + name + '=======================')
                json_2_img(json_path, save_path)

    all_json_2_img("leaf-master/data/femnist/3500_client_json/", "leaf-master/data/femnist/3500_client_img/")
    ```

    If the following information is displayed, the script is successfully executed:

    ```sh
    =============================f0644_19.json=======================
    ----- 0 -----
    ----- 1 -----
    ----- 2 -----
    ......
    ```

7. Because the data set in some user folders is small, if the number is smaller than the batch size, random expansion is required.

    You can refer to the following code to check and expand the entire data set `"leaf/data/femnist/3500_client_img/"`:

    ```python
    import os
    import shutil
    from random import choice

    def count_dir(path):
        num = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                num += 1
        return num

    def get_img_list(path):
        img_path_list = []
        label_list = os.listdir(path)
        for i in range(len(label_list)):
            label = label_list[i]
            imgs_path = os.path.join(path, label)
            imgs_name = os.listdir(imgs_path)
            for j in range(len(imgs_name)):
                img_name = imgs_name[j]
                img_path = os.path.join(imgs_path, img_name)
                img_path_list.append(img_path)
        return img_path_list

    def data_aug(data_root_path, batch_size = 32):
        users = os.listdir(data_root_path)
        tags = ["train", "test"]
        aug_users = []
        for i in range(len(users)):
            user = users[i]
            for tag in tags:
                data_path = os.path.join(data_root_path, user, tag)
                num_data = count_dir(data_path)
                if num_data < batch_size:
                    aug_users.append(user + "_" + tag)
                    print("user: ", user, " ", tag, " data number: ", num_data, " < ", batch_size, " should be aug")
                    aug_num = batch_size - num_data
                    img_path_list = get_img_list(data_path)
                    for j in range(aug_num):
                        img_path = choice(img_path_list)
                        info = img_path.split(".")
                        aug_img_path = info[0] + "_aug_" + str(j) + ".png"
                        shutil.copy(img_path, aug_img_path)
                        print("[aug", j, "]", "============= copy file:", img_path, "to ->", aug_img_path)
        print("the number of all aug users: " + str(len(aug_users)))
        print("aug user name: ", end=" ")
        for k in range(len(aug_users)):
            print(aug_users[k], end = " ")

    if __name__ == "__main__":
        data_root_path = "leaf/data/femnist/3500_client_img/"
        batch_size = 32
        data_aug(data_root_path,  batch_size)
    ```

8. Convert the expanded image dataset into a .bin file format available to the federated learning framework.

    For details, see the following code:

    ```python
    import numpy as np
    import os
    import mindspore.dataset as ds
    import mindspore.dataset.transforms.c_transforms as tC
    import mindspore.dataset.vision.py_transforms as PV
    import mindspore.dataset.transforms.py_transforms as PT
    import mindspore

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    def count_id(path):
        files = os.listdir(path)
        ids = {}
        for i in files:
            ids[i] = int(i)
        return ids

    def create_dataset_from_folder(data_path, img_size, batch_size=32, repeat_size=1, num_parallel_workers=1, shuffle=False):
        """ create dataset for train or test
            Args:
                data_path: Data path
                batch_size: The number of data records in each group
                repeat_size: The number of replicated data records
                num_parallel_workers: The number of parallel workers
            """
        # define dataset
        ids = count_id(data_path)
        mnist_ds = ds.ImageFolderDataset(dataset_dir=data_path, decode=False, class_indexing=ids)
        # define operation parameters
        resize_height, resize_width = img_size[0], img_size[1]  # 32

        transform = [
            PV.Decode(),
            PV.Grayscale(1),
            PV.Resize(size=(resize_height, resize_width)),
            PV.Grayscale(3),
            PV.ToTensor(),
        ]
        compose = PT.Compose(transform)

        # apply map operations on images
        mnist_ds = mnist_ds.map(input_columns="label", operations=tC.TypeCast(mindspore.int32))
        mnist_ds = mnist_ds.map(input_columns="image", operations=compose)

        # apply DatasetOps
        buffer_size = 10000
        if shuffle:
            mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
        mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
        mnist_ds = mnist_ds.repeat(repeat_size)
        return mnist_ds

    def img2bin(root_path, root_save):
        """
        transform images to bin files

        Args:
        root_path: the root path of 3500 images files
        root_save: the root path to save bin files

        """

        use_list = []
        train_batch_num = []
        test_batch_num = []
        mkdir(root_save)
        users = os.listdir(root_path)
        for user in users:
            use_list.append(user)
            user_path = os.path.join(root_path, user)
            train_test = os.listdir(user_path)
            for tag in train_test:
                data_path = os.path.join(user_path, tag)
                dataset = create_dataset_from_folder(data_path, (32, 32, 1), 32)
                batch_num = 0
                img_list = []
                label_list = []
                for data in dataset.create_dict_iterator():
                    batch_x_tensor = data['image']
                    batch_y_tensor = data['label']
                    trans_img = np.transpose(batch_x_tensor.asnumpy(), [0, 2, 3, 1])
                    img_list.append(trans_img)
                    label_list.append(batch_y_tensor.asnumpy())
                    batch_num += 1

                if tag == "train":
                    train_batch_num.append(batch_num)
                elif tag == "test":
                    test_batch_num.append(batch_num)

                imgs = np.array(img_list)  # (batch_num, 32,3,32,32)
                labels = np.array(label_list)
                path1 = os.path.join(root_save, user)
                mkdir(path1)
                image_path = os.path.join(path1, user + "_" + "bn_" + str(batch_num) + "_" + tag + "_data.bin")
                label_path = os.path.join(path1, user + "_" + "bn_" + str(batch_num) + "_" + tag + "_label.bin")

                imgs.tofile(image_path)
                labels.tofile(label_path)
                print("user: " + user + " " + tag + "_batch_num: " + str(batch_num))
        print("total " + str(len(use_list)) + " users finished!")

    root_path = "leaf-master/data/femnist/3500_client_img/"
    root_save = "leaf-master/data/femnist/3500_clients_bin"
    img2bin(root_path, root_save)
    ```

    If the following information is displayed, the script is successfully executed:

    ```sh
    user: f0141_43 test_batch_num: 1
    user: f0141_43 train_batch_num: 10
    user: f0137_14 test_batch_num: 1
    user: f0137_14 train_batch_num: 11
    ......
    total 3500 users finished!
    ```

9. The 3500_clients_bin folder contains 3500 user folders. The directory structure is as follows:

    ```sh
    leaf-master/data/femnist/3500_clients_bin
    ├── f0000_14  # User ID.
    │   ├── f0000_14_bn_10_train_data.bin  # Training data of user f0000_14 (The number 10 following bn_ indicates the batch number.)
    │   ├── f0000_14_bn_10_train_label.bin  # Training label of user f0000_14.
    │   ├── f0000_14_bn_1_test_data.bin  # Test data of user f0000_14 (The number 1 following bn_ indicates the batch number.)
    │   └── f0000_14_bn_1_test_label.bin  # Training label of user f0000_14.
    ├── f0001_41  # User ID.
    │   ├── f0001_41_bn_11_train_data.bin  # Training data of user f0001_41 (The number 11 following bn_ indicates the batch number.)
    │   ├── f0001_41_bn_11_train_label.bin  # Training label of user f0001_41.
    │   ├── f0001_41_bn_1_test_data.bin  # Test data of user f0001_41 (The number 1 following bn_ indicates the batch number.)
    │   └── f0001_41_bn_1_test_label.bin  # Test label of user f0001_41.
    │
    │
    │                    ...
    │
    │
    └── f4099_10  # User ID.
        ├── f4099_10_bn_4_train_data.bin  # Training data of user f4099_10 (The number 4 following bn_ indicates the batch number.)
        ├── f4099_10_bn_4_train_label.bin  # Training label of user f4099_10.
        ├── f4099_10_bn_1_test_data.bin  # Test data of user f4099_10 (The number 1 following bn_ indicates the batch number.)
        └── f4099_10_bn_1_test_label.bin  # Test label of user f4099_10.
    ```

The `3500_clients_bin` folder generated according to the above steps 1-9 can be directly used as the input data of the Image Classification task for Cross-device Federated Learning.