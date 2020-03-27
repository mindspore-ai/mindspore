# Copyright 2020 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# import jsbeautifier

import os
import urllib
import urllib.request


def create_data_cache_dir():
    cwd = os.getcwd()
    target_directory = os.path.join(cwd, "data_cache")
    try:
        if not (os.path.exists(target_directory)):
            os.mkdir(target_directory)
    except OSError:
        print("Creation of the directory %s failed" % target_directory)
    return target_directory;


def download_and_uncompress(files, source_url, target_directory, is_tar=False):
    for f in files:
        url = source_url + f
        target_file = os.path.join(target_directory, f)

        ##check if file already downloaded
        if not (os.path.exists(target_file) or os.path.exists(target_file[:-3])):
            urllib.request.urlretrieve(url, target_file)
            if is_tar:
                print("extracting from local tar file " + target_file)
                rc = os.system("tar -C " + target_directory + " -xvf " + target_file)
            else:
                print("unzipping " + target_file)
                rc = os.system("gunzip -f " + target_file)
            if rc != 0:
                print("Failed to uncompress ", target_file, " removing")
                os.system("rm " + target_file)
                ##exit with error so that build script will fail
                raise SystemError
        else:
            print("Using cached dataset at ", target_file)


def download_mnist(target_directory=None):
    if target_directory == None:
        target_directory = create_data_cache_dir()

        ##create mnst directory
        target_directory = os.path.join(target_directory, "mnist")
        try:
            if not (os.path.exists(target_directory)):
                os.mkdir(target_directory)
        except OSError:
            print("Creation of the directory %s failed" % target_directory)

    MNIST_URL = "http://yann.lecun.com/exdb/mnist/"
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']
    download_and_uncompress(files, MNIST_URL, target_directory, is_tar=False)

    return target_directory, os.path.join(target_directory, "datasetSchema.json")


CIFAR_URL = "https://www.cs.toronto.edu/~kriz/"


def download_cifar(target_directory, files, directory_from_tar):
    if target_directory == None:
        target_directory = create_data_cache_dir()

    download_and_uncompress([files], CIFAR_URL, target_directory, is_tar=True)

    ##if target dir was specify move data from directory created by tar
    ##and put data into target dir
    if target_directory != None:
        tar_dir_full_path = os.path.join(target_directory, directory_from_tar)
        all_files = os.path.join(tar_dir_full_path, "*")
        cmd = "mv " + all_files + " " + target_directory
        if os.path.exists(tar_dir_full_path):
            print("copy files back to target_directory")
            print("Executing: ", cmd)
            rc1 = os.system(cmd)
            rc2 = os.system("rm -r " + tar_dir_full_path)
            if rc1 != 0 or rc2 != 0:
                print("error when running command: ", cmd)
                download_file = os.path.join(target_directory, files)
                print("removing " + download_file)
                os.system("rm " + download_file)

                ##exit with error so that build script will fail
                raise SystemError

    ##change target directory to directory after tar
    return os.path.join(target_directory, directory_from_tar)


def download_cifar10(target_directory=None):
    return download_cifar(target_directory, "cifar-10-binary.tar.gz", "cifar-10-batches-bin")


def download_cifar100(target_directory=None):
    return download_cifar(target_directory, "cifar-100-binary.tar.gz", "cifar-100-binary")


def download_all_for_test(cwd):
    download_mnist(os.path.join(cwd, "testMnistData"))


##Download all datasets to existing test directories
if __name__ == "__main__":
    download_all_for_test(os.getcwd())
