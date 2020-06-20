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
"""Built-in iterators.
"""
from abc import abstractmethod
import copy
import weakref

from mindspore._c_dataengine import DEPipeline
from mindspore._c_dataengine import OpName

from mindspore import log as logger
from . import datasets as de


ITERATORS_LIST = list()


def _cleanup():
    """Release all the Iterator."""
    for itr_ref in ITERATORS_LIST:
        itr = itr_ref()
        if itr is not None:
            itr.release()


def alter_tree(node):
    """Traversing the python Dataset tree/graph to perform some alteration to some specific nodes."""
    if not node.children:
        return _alter_node(node)

    converted_children = []
    for input_op in node.children:
        converted_children.append(alter_tree(input_op))
    node.children = converted_children
    return _alter_node(node)


def _alter_node(node):
    """DEPRECATED"""
    # Please check ccsrc/dataset/engine/opt for tree transformation.
    if isinstance(node, de.MapDataset):
        if node.python_multiprocessing:
            # Bootstrap can only be performed on a copy of the original dataset node.
            # Bootstrap on original dataset node will make all iterators share the same process pool
            node.iterator_bootstrap()
    return node


class Iterator:
    """
    General Iterator over a dataset.

    Attributes:
        dataset: Dataset to be iterated over
    """

    def __init__(self, dataset):
        ITERATORS_LIST.append(weakref.ref(self))
        # create a copy of tree and work on it.
        self.dataset = copy.deepcopy(dataset)
        self.dataset = alter_tree(self.dataset)
        if not self.__is_tree():
            raise ValueError("The data pipeline is not a tree (i.e., one node has 2 consumers)")
        self.depipeline = DEPipeline()

        # for manifest temporary use
        self.__batch_node(self.dataset, 0)

        root = self.__convert_node_postorder(self.dataset)
        self.depipeline.AssignRootNode(root)
        self.depipeline.LaunchTreeExec()
        self._index = 0

    def __is_tree_node(self, node):
        """Check if a node is tree node."""
        if not node.children:
            if len(node.parent) > 1:
                return False

        if len(node.parent) > 1:
            return False

        for input_node in node.children:
            cls = self.__is_tree_node(input_node)
            if not cls:
                return False
        return True

    def __is_tree(self):
        return self.__is_tree_node(self.dataset)

    @staticmethod
    def __get_dataset_type(dataset):
        """Get the dataset type."""
        op_type = None
        if isinstance(dataset, de.ShuffleDataset):
            op_type = OpName.SHUFFLE
        elif isinstance(dataset, de.MindDataset):
            op_type = OpName.MINDRECORD
        elif isinstance(dataset, de.BatchDataset):
            op_type = OpName.BATCH
        elif isinstance(dataset, de.BucketBatchByLengthDataset):
            op_type = OpName.BUCKETBATCH
        elif isinstance(dataset, de.SyncWaitDataset):
            op_type = OpName.BARRIER
        elif isinstance(dataset, de.ZipDataset):
            op_type = OpName.ZIP
        elif isinstance(dataset, de.ConcatDataset):
            op_type = OpName.CONCAT
        elif isinstance(dataset, de.MapDataset):
            op_type = OpName.MAP
        elif isinstance(dataset, de.FilterDataset):
            op_type = OpName.FILTER
        elif isinstance(dataset, de.RepeatDataset):
            op_type = OpName.REPEAT
        elif isinstance(dataset, de.SkipDataset):
            op_type = OpName.SKIP
        elif isinstance(dataset, de.TakeDataset):
            op_type = OpName.TAKE
        elif isinstance(dataset, de.ImageFolderDatasetV2):
            op_type = OpName.IMAGEFOLDER
        elif isinstance(dataset, de.GeneratorDataset):
            op_type = OpName.GENERATOR
        elif isinstance(dataset, de.TransferDataset):
            op_type = OpName.DEVICEQUEUE
        elif isinstance(dataset, de.RenameDataset):
            op_type = OpName.RENAME
        elif isinstance(dataset, de.TFRecordDataset):
            op_type = OpName.TFREADER
        elif isinstance(dataset, de.ProjectDataset):
            op_type = OpName.PROJECT
        elif isinstance(dataset, de.MnistDataset):
            op_type = OpName.MNIST
        elif isinstance(dataset, de.ManifestDataset):
            op_type = OpName.MANIFEST
        elif isinstance(dataset, de.VOCDataset):
            op_type = OpName.VOC
        elif isinstance(dataset, de.CocoDataset):
            op_type = OpName.COCO
        elif isinstance(dataset, de.Cifar10Dataset):
            op_type = OpName.CIFAR10
        elif isinstance(dataset, de.Cifar100Dataset):
            op_type = OpName.CIFAR100
        elif isinstance(dataset, de.CelebADataset):
            op_type = OpName.CELEBA
        elif isinstance(dataset, de.RandomDataset):
            op_type = OpName.RANDOMDATA
        elif isinstance(dataset, de.TextFileDataset):
            op_type = OpName.TEXTFILE
        elif isinstance(dataset, de.BuildVocabDataset):
            op_type = OpName.BUILDVOCAB
        elif isinstance(dataset, de.CLUEDataset):
            op_type = OpName.CLUE
        else:
            raise ValueError("Unsupported DatasetOp")

        return op_type

    # Convert python node into C node and add to C layer execution tree in postorder traversal.
    def __convert_node_postorder(self, node):
        op_type = self.__get_dataset_type(node)
        c_node = self.depipeline.AddNodeToTree(op_type, node.get_args())

        for py_child in node.children:
            c_child = self.__convert_node_postorder(py_child)
            self.depipeline.AddChildToParentNode(c_child, c_node)

        return c_node

    def __batch_node(self, dataset, level):
        """Recursively get batch node in the dataset tree."""
        if isinstance(dataset, de.BatchDataset):
            return
        for input_op in dataset.children:
            self.__batch_node(input_op, level + 1)

    @staticmethod
    def __print_local(dataset, level):
        """Recursively print the name and address of nodes in the dataset tree."""
        name = dataset.__class__.__name__
        ptr = hex(id(dataset))
        for _ in range(level):
            logger.info("\t", end='')
        if not dataset.children:
            logger.info("-%s (%s)", name, ptr)
        else:
            logger.info("+%s (%s)", name, ptr)
        for input_op in dataset.children:
            Iterator.__print_local(input_op, level + 1)

    def print(self):
        """Print the dataset tree"""
        self.__print_local(self.dataset, 0)

    def release(self):
        if hasattr(self, 'depipeline') and self.depipeline:
            del self.depipeline

    @abstractmethod
    def get_next(self):
        pass

    def __next__(self):
        data = self.get_next()
        if not data:
            if self._index == 0:
                logger.warning("No records available.")
            raise StopIteration
        self._index += 1
        return data

    def get_output_shapes(self):
        return [t for t in self.depipeline.GetOutputShapes()]

    def get_output_types(self):
        return [t for t in self.depipeline.GetOutputTypes()]

    def get_dataset_size(self):
        return self.depipeline.GetDatasetSize()

    def get_batch_size(self):
        return self.depipeline.GetBatchSize()

    def get_repeat_count(self):
        return self.depipeline.GetRepeatCount()

    def num_classes(self):
        return self.depipeline.GetNumClasses()

    def __deepcopy__(self, memo):
        return self


class DictIterator(Iterator):
    """
    The derived class of Iterator with dict type.
    """

    def __iter__(self):
        return self

    def get_next(self):
        """
        Returns the next record in the dataset as dictionary

        Returns:
            Dict, the next record in the dataset.
        """

        return {k: v.as_array() for k, v in self.depipeline.GetNextAsMap().items()}


class TupleIterator(Iterator):
    """
    The derived class of Iterator with list type.
    """

    def __init__(self, dataset, columns=None):
        if columns is not None:
            if not isinstance(columns, list):
                columns = [columns]
            dataset = dataset.project(columns)
        super().__init__(dataset)

    def __iter__(self):
        return self

    def get_next(self):
        """
        Returns the next record in the dataset as a list

        Returns:
            List, the next record in the dataset.
        """

        return [t.as_array() for t in self.depipeline.GetNextAsList()]
