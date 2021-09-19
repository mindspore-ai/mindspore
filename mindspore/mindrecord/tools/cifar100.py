# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Cifar100 reader class.
"""
import builtins
import io
import pickle
import os
import numpy as np

from ..shardutils import check_filename

__all__ = ['Cifar100']

safe_builtins = {
    'range',
    'complex',
    'set',
    'frozenset',
    'slice',
}


class RestrictedUnpickler(pickle.Unpickler):
    """
    Unpickle allowing only few safe classes from the builtins module or numpy

    Raises:
        pickle.UnpicklingError: If there is a problem unpickling an object
    """
    def find_class(self, module, name):
        # Only allow safe classes from builtins and numpy
        if module == "builtins" and name in safe_builtins:
            return getattr(builtins, name)
        if module == "numpy.core.multiarray" and name == "_reconstruct":
            return getattr(np.core.multiarray, name)
        if module == "numpy":
            return getattr(np, name)
        # Forbid everything else.
        raise pickle.UnpicklingError("global '%s.%s' is forbidden" % (module, name))



def restricted_loads(s):
    """Helper function analogous to pickle.loads()."""
    if isinstance(s, str):
        raise TypeError("can not load pickle from unicode string")
    f = io.BytesIO(s)
    try:
        return RestrictedUnpickler(f, encoding='bytes').load()
    except pickle.UnpicklingError:
        raise RuntimeError("Not a valid Cifar100 Dataset.")
    except UnicodeDecodeError:
        raise RuntimeError("Not a valid Cifar100 Dataset.")
    except Exception:
        raise RuntimeError("Unexpected error while Unpickling Cifar100 Dataset.")


class Cifar100:
    """
    Class to convert cifar100 to MindRecord.

    Cifar100 contains train & test data. There are 500 training images and
    100 testing images per class. The 100 classes in the CIFAR-100 are grouped
    into 20 superclasses.

    Args:
        path (str): cifar100 directory which contain train and test binary data.
        one_hot (bool): one_hot flag.
    """
    class Test:
        pass

    def __init__(self, path, one_hot=True):
        check_filename(path)
        self.path = path
        if not isinstance(one_hot, bool):
            raise ValueError("The parameter one_hot must be bool")
        self.one_hot = one_hot
        self.images = None
        self.fine_labels = None
        self.coarse_labels = None

    def load_data(self):
        """
        Returns a list which contain train data & two labels, test data & two labels.

        Returns:
            list, train and test images, fine labels, coarse labels.
        """
        dic = {}
        fine_labels = []
        coarse_labels = []
        files = os.listdir(self.path)
        for file in files:
            if file == "train":
                real_file_path = os.path.realpath(self.path)
                with open(os.path.join(real_file_path, file), 'rb') as f: # load train data
                    dic = restricted_loads(f.read())
                    images = np.array(dic[b"data"].reshape([-1, 3, 32, 32]))
                    fine_labels.append(dic[b"fine_labels"])
                    coarse_labels.append(dic[b"coarse_labels"])
            elif file == "test":                                     # load test data
                real_file_path = os.path.realpath(self.path)
                with open(os.path.join(real_file_path, file), 'rb') as f:
                    dic = restricted_loads(f.read())
                    test_images = np.array(dic[b"data"].reshape([-1, 3, 32, 32]))
                    test_fine_labels = np.array(dic[b"fine_labels"])
                    test_coarse_labels = np.array(dic[b"coarse_labels"])
        dic["train_images"] = images.transpose(0, 2, 3, 1)
        dic["train_fine_labels"] = np.array(fine_labels).reshape([-1, 1])
        dic["train_coarse_labels"] = np.array(coarse_labels).reshape([-1, 1])
        dic["test_images"] = test_images.transpose(0, 2, 3, 1)
        dic["test_fine_labels"] = test_fine_labels.reshape([-1, 1])
        dic["test_coarse_labels"] = test_coarse_labels.reshape([-1, 1])
        if self.one_hot:
            dic["train_fine_labels"] = self._one_hot(dic["train_fine_labels"], 100)
            dic["train_coarse_labels"] = self._one_hot(dic["train_coarse_labels"], 20)
            dic["test_fine_labels"] = self._one_hot(dic["test_fine_labels"], 100)
            dic["test_coarse_labels"] = self._one_hot(dic["test_coarse_labels"], 20)

        self.images, self.fine_labels, self.coarse_labels = \
            dic["train_images"], dic["train_fine_labels"], dic["train_coarse_labels"]
        self.Test.images, self.Test.fine_labels, self.Test.coarse_labels = \
            dic["test_images"], dic["test_fine_labels"], dic["test_coarse_labels"]
        return [dic["train_images"], dic["train_fine_labels"], dic["train_coarse_labels"],
                dic["test_images"], dic["test_fine_labels"], dic["test_coarse_labels"]]

    def _one_hot(self, labels, num):
        """
        Returns a numpy.

        Returns:
            Object, numpy array.
        """
        size = labels.shape[0]
        label_one_hot = np.zeros([size, num])
        for i in range(size):
            label_one_hot[i, np.squeeze(labels[i])] = 1
        return label_one_hot
