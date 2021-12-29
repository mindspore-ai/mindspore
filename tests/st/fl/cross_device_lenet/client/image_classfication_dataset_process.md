# 联邦学习图像分类数据集处理

本教程采用`leaf`数据集中的联邦学习数据集`FEMNIST`， 该数据集包含62个不同类别的手写数字和字母（数字0~9、26个小写字母、26个大写字母），图像大小为`28 x 28`像素，数据集包含3500个用户的手写数字和字母（最多可模拟3500个客户端参与联邦学习），总数据量为805263，平均每个用户包含数据量为226.83，所有用户数据量的方差为88.94。

## 端云联邦学习图像分类数据集处理

参考[leaf数据集官方指导](https://github.com/TalwalkarLab/leaf)下载数据集。

1. 下载数据集前的环境要求。

    ```sh
    numpy==1.16.4
    scipy                      # conda install scipy
    tensorflow==1.13.1         # pip install tensorflow
    Pillow                     # pip install Pillow
    matplotlib                 # pip install matplotlib
    jupyter                    # conda install jupyter notebook==5.7.8 tornado==4.5.3
    pandas                     # pip install pandas
    ```

2. 使用git下载官方数据集生成脚本。

    ```sh
    git clone https://github.com/TalwalkarLab/leaf.git
    ```

    下载项目后，目录结构如下：

    ```sh
    leaf/data/femnist
        ├── data  # 用来存放指令生成的数据集
        ├── preprocess  # 存放数据预处理的相关代码
        ├── preprocess.sh  # femnist数据集生成shell脚本
        └── README.md  # 官方数据集下载指导文档
    ```

3. 以`femnist`数据集为例，运行以下指令进入指定路径。

    ```sh
    cd  leaf/data/femnist
    ```

4. 用指令`./preprocess.sh -s niid --sf 1.0 -k 0 -t sample`生成的数据集包含3500个用户，且按照9:1对每个用户的数据划分训练和测试集。

    指令中参数含义可参考`leaf/data/femnist/README.md`文件中的说明。

    运行之后目录结构如下：

    ```text
    leaf/data/femnist/35_client_sf1_data/
        ├── all_data  # 所有数据集混合在一起，不区分训练测试集，共包含35个json文件，每个json文件包含100个用户的数据
        ├── test  # 按照9:1对每个用户的数据划分训练和测试集后的测试集，共包含35个json文件，每个json文件包含100个用户的数据
        ├── train  # 按照9:1对每个用户的数据划分训练和测试集后的训练集，共包含35个json文件，每个json文件包含100个用户的数据
        └── ...  # 其他文件，暂不需要用到，不作介绍
    ```

    其中每个json文件包含以下三个部分：

    - `users`: 用户列表。
    - `num_samples`: 每个用户的样本数量列表。
    - `user_data`: 一个以用户名为key，以它们各自的数据为value的字典对象；对于每个用户，数据表示为图像列表，每张图像表示为大小为784的整数列表（将`28 x 28`图像数组展平所得）。

    在重新运行`preprocess.sh`之前，请确保删除数据目录中的`rem_user_data`、`sampled_data`、`test`和`train`子文件夹。

5. 将35个json文件划分为3500个json文件（每个json文件代表一个用户）。

    参考代码如下：

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
    partition_json("leaf/data/femnist/35_client_sf1_data/train", "leaf/data/femnist/3500_client_json/train")
    # partition test json files
    partition_json("leaf/data/femnist/35_client_sf1_data/test", "leaf/data/femnist/3500_client_json/test")
    ```

    其中`root_path`为`leaf/data/femnist/35_client_sf1_data/{train,test}`，`new_root_path`自行设置，用于存放生成的3500个用户json文件，需分别对训练和测试文件夹进行处理。

    新生成的3500个用户json文件，每个文件均包含以下三个部分：

    - `user_name`: 用户名。
    - `num_samples`: 用户的样本数。
    - `user_data`: 一个以'x'为key，以用户数据为value的字典对象； 以'y'为key，以用户数据对应的标签为value。

    运行该脚本打印如下，代表运行成功：

    ```sh
    ======== process 1 file: /leaf/data/femnist/35_client_sf1_data/train/all_data_16_niid_0_keep_0_train_9.json======================
    ---processing user: 1---
    ---processing user: 2---
    ---processing user: 3---
    ......
    ```

6. 将json文件转换为图片文件。

    可参考如下代码：

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

    all_json_2_img("leaf/data/femnist/3500_client_json/", "leaf/data/femnist/3500_client_img/")
    ```

    运行该脚本打印如下，代表运行成功：

    ```sh
    =============================f0644_19.json=======================
    ----- 0 -----
    ----- 1 -----
    ----- 2 -----
    ......
    ```

7. 由于有些用户文件夹下的数据集较小，若数量小于batch size，需要进行随机扩充。

    可参考下面代码对整个数据集`"leaf/data/femnist/3500_client_img/"`进行检查并扩充：

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

8. 将扩充后图片数据集转换为联邦学习框架可用的bin文件格式。

    可参考下面代码：

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

    root_path = "leaf/data/femnist/3500_client_img/"
    root_save = "leaf/data/femnist/3500_clients_bin"
    img2bin(root_path, root_save)
    ```

    运行该脚本打印如下，代表运行成功：

    ```sh
    user: f0141_43 test_batch_num: 1
    user: f0141_43 train_batch_num: 10
    user: f0137_14 test_batch_num: 1
    user: f0137_14 train_batch_num: 11
    ......
    total 3500 users finished!
    ```

9. 生成`3500_clients_bin`文件夹内共包含3500个用户文件夹，其目录结构如下：

    ```sh
    leaf/data/femnist/3500_clients_bin
      ├── f0000_14  # 用户编号
      │   ├── f0000_14_bn_10_train_data.bin  # 用户f0000_14的训练数据 （bn_后面的数字10代表batch number）
      │   ├── f0000_14_bn_10_train_label.bin  # 用户f0000_14的训练标签
      │   ├── f0000_14_bn_1_test_data.bin  # 用户f0000_14的测试数据   （bn_后面的数字1代表batch number）
      │   └── f0000_14_bn_1_test_label.bin  # 用户f0000_14的测试标签
      ├── f0001_41  # 用户编号
      │   ├── f0001_41_bn_11_train_data.bin  # 用户f0001_41的训练数据 （bn_后面的数字11代表batch number）
      │   ├── f0001_41_bn_11_train_label.bin  # 用户f0001_41的训练标签
      │   ├── f0001_41_bn_1_test_data.bin  # 用户f0001_41的测试数据   （bn_后面的数字1代表batch number）
      │   └── f0001_41_bn_1_test_label.bin  #  用户f0001_41的测试标签
      │                    ...
      └── f4099_10  # 用户编号
          ├── f4099_10_bn_4_train_data.bin  # 用户f4099_10的训练数据 （bn_后面的数字4代表batch number）
          ├── f4099_10_bn_4_train_label.bin  # 用户f4099_10的训练标签
          ├── f4099_10_bn_1_test_data.bin  # 用户f4099_10的测试数据   （bn_后面的数字1代表batch number）
          └── f4099_10_bn_1_test_label.bin  # 用户f4099_10的测试标签
    ```

根据以上1～9步骤生成的`3500_clients_bin`文件夹可直接作为端云联邦图像分类任务的输入数据。

