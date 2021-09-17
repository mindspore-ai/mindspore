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
Cifar10 reader class.
"""
import builtins
import io
import pickle
import re
import os
import numpy as np

from ..shardutils import check_filename

__all__ = ['Cifar10']

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
        raise RuntimeError("Not a valid Cifar10 Dataset.")
    except UnicodeDecodeError:
        raise RuntimeError("Not a valid Cifar10 Dataset.")
    except Exception:
        raise RuntimeError("Unexpected error while Unpickling Cifar10 Dataset.")


class Cifar10:
    """
    Class to convert cifar10 to MindRecord.

    Args:
        path (str): cifar10 directory which contain data_batch_* and test_batch.
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
        self.labels = None

    def load_data(self):
        """
        Returns a list which contain data & label, test & label.

        Returns:
            list, train images, train labels and test images, test labels
        """
        dic = {}
        images = np.zeros([10000, 3, 32, 32])
        labels = []
        files = os.listdir(self.path)
        for file in files:
            if re.match("data_batch_*", file):
                real_file_path = os.path.realpath(self.path)
                with open(os.path.join(real_file_path, file), 'rb') as f: # load train data
                    dic = restricted_loads(f.read())
                    images = np.r_[images, dic[b"data"].reshape([-1, 3, 32, 32])]
                    labels.append(dic[b"labels"])
            elif re.match("test_batch", file):                       # load test data
                real_file_path = os.path.realpath(self.path)
                with open(os.path.join(real_file_path, file), 'rb') as f:
                    dic = restricted_loads(f.read())
                    test_images = np.array(dic[b"data"].reshape([-1, 3, 32, 32]))
                    test_labels = np.array(dic[b"labels"])
        dic["train_images"] = images[10000:].transpose(0, 2, 3, 1)
        dic["train_labels"] = np.array(labels).reshape([-1, 1])
        dic["test_images"] = test_images.transpose(0, 2, 3, 1)
        dic["test_labels"] = test_labels.reshape([-1, 1])
        if self.one_hot:
            dic["train_labels"] = self._one_hot(dic["train_labels"], 10)
            dic["test_labels"] = self._one_hot(dic["test_labels"], 10)

        self.images, self.labels = dic["train_images"], dic["train_labels"]
        self.Test.images, self.Test.labels = dic["test_images"], dic["test_labels"]
        return [dic["train_images"], dic["train_labels"], dic["test_images"], dic["test_labels"]]

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
