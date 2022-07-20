"""
This code mainly explains how to build the inference model of mindspore-lite
through the Python api, and at the same time verifies the inference accuracy of
the specified model in the ILSVRC2012-ImageNet dataset.
"""

import argparse
from os import listdir
from os import path
import numpy as np
import cv2
from tqdm import tqdm
import mindspore_lite as mslite


def get_label_map(label_file):
    with open(label_file) as ground_truth:
        lines = ground_truth.readlines()
        label_dict = {}
        for index, line in enumerate(lines):
            label_dict[line[:-1]] = index
    return label_dict


label_map = get_label_map("synsets.txt")


class DataSet:
    """
    For loading Validation data from the ImageNet DataSet.
    """

    def __init__(self, images_dir):
        self.images_dir = images_dir
        self.img_ids = listdir(images_dir)
        self.img_ids.sort(key=lambda x: x.split('.')[0])

    @staticmethod
    def normalize(img_array):
        """Normalize the current image data"""
        assert isinstance(img_array, np.ndarray) and img_array.shape == (
            3, 224, 224), "please check img_array's type and shape "
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_img = np.zeros(img_array.shape).astype('float32')
        for i in range(img_array.shape[0]):
            # for each pixel in each channel, divide the value by 255
            # to get value between [0, 1] and then normalize
            norm_img[i, :, :] = (img_array[i, :, :]/255 -
                                 mean_vec[i]) / stddev_vec[i]
        return norm_img

    def len(self):
        """Get the number of samples in the current folder"""
        return len(self.img_ids)

    def getitem(self, index):
        """Process the index-th image data in the current folder"""
        img_name = self.img_ids[index]
        img_path = self.images_dir + '/' + img_name
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img_array = np.array(img, dtype=np.float32).transpose(2, 0, 1)
        norm_img = self.normalize(img_array)
        norm_img = norm_img.transpose(1, 2, 0)
        norm_img = norm_img[np.newaxis, :]
        return norm_img


def create_model(model_path):
    assert path.isfile(model_path), "Please make sure your model file exists"
    cpu_device_info = mslite.CPUDeviceInfo(enable_fp16=False)
    print("cpu_device_info: ", cpu_device_info)
    context = mslite.Context(thread_num=1, thread_affinity_mode=2)
    context.append_device_info(cpu_device_info)
    model = mslite.Model()
    model.build_from_file(model_path, mslite.ModelType.MINDIR_LITE, context)
    return model


def inference(input_model, input_data):
    # process input for inference
    model = input_model
    inputs = model.get_inputs()
    inputs[0].set_data_from_numpy(input_data)
    outputs = model.get_outputs()
    model.predict(inputs, outputs)
    data = outputs[0].get_data_to_numpy()
    return data


def post_process(output_data):
    # Get the index of the maximum value of the output data ,which is the
    # prediction class
    res_index = output_data.argmax(axis=1)
    return res_index


def get_args():
    """Get command line arguments"""
    parser = argparse.ArgumentParser(
        description='Used to verify the inference accuracy of common models\
                     on ImageNet using Mindspore-Lite')
    parser.add_argument('--dataset_dir', '-d', type=str,
                        required=True,
                        help="Path to a directory containing ImageNet dataset.\n\
                              This folder should contain the val subfolder and\
                              the label file synsets.txt")
    parser.add_argument('--model', '-m', type=str,
                        required=True,
                        help='The mindspore-lite model file for inference')
    parser.add_argument('--num_of_cls', '-c', type=int,
                        default=100,
                        help='Number of classes to use ,\
                              Your input must between 1 to 1000')
    parser.add_argument('--num_per_cls', '-n', type=int,
                        default=1,
                        help='Number of samples to use per class,\
                        Your input must between 1 to 50')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    inference_model = create_model(args.model)
    imagnet_val_dir = path.join(args.dataset_dir, 'val')
    cls_ids = listdir(imagnet_val_dir)
    # pylint: disable=C0103
    accuracy = 0.0
    # get the cls_ids to participate in verification
    cls_ids = cls_ids[0:args.num_of_cls]
    for cls_id in tqdm(cls_ids, ncols=80):
        try:
            cls_label = label_map[cls_id]
        except KeyError:
            raise KeyError("cls_id {0} not in label_map.keys()".format(cls_id))
        image_dir = path.join(imagnet_val_dir, cls_id)
        img_dataset = DataSet(image_dir)
        # get the number of samples to use per class
        cur_data_len = args.num_per_cls \
            if args.num_per_cls < img_dataset.len()else img_dataset.len()

        for idx in range(cur_data_len):
            inference_data = img_dataset.getitem(idx)
            res_data = inference(inference_model, inference_data)
            res_label = post_process(res_data)
            accuracy += (res_label == cls_label)

    accuracy = accuracy / (args.num_of_cls * args.num_per_cls)
    print("When using {0} classes with {1} samples each,the inference accuracy \
    is {2}".format(args.num_of_cls, args.num_per_cls, accuracy))
