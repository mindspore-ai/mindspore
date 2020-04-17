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
    for itr_ref in ITERATORS_LIST:
        itr = itr_ref()
        if itr is not None:
            itr.release()


def alter_tree(node):
    """Traversing the python Dataset tree/graph to perform some alteration to some specific nodes."""
    if not node.input:
        return _alter_node(node)

    converted_children = []
    for input_op in node.input:
        converted_children.append(alter_tree(input_op))
    node.input = converted_children
    return _alter_node(node)


def _alter_node(node):
    """Performing some alteration to a dataset node. A common alteration is to insert a node."""
    if isinstance(node, de.TFRecordDataset) and node.shuffle_level == de.Shuffle.GLOBAL:
        # Remove the connection between the parent's node to the current node because we are inserting a node.
        if node.output:
            node.output.pop()
        # Perform a fast scan for average rows per file
        avg_rows_per_file = node.get_dataset_size(True) // len(node.dataset_files)
        # Shuffle between 4 files with a minimum size of 10000 rows
        new_shuffle = node.shuffle(max(avg_rows_per_file * 4, 10000))
        return new_shuffle

    if isinstance(node, de.MapDataset):
        if node.columns_order is not None:
            # Remove the connection between the parent's node to the current node because we are inserting a node.
            if node.output:
                node.output.pop()

            return node.project(node.columns_order)
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
        if not node.input:
            if len(node.output) > 1:
                return False

        if len(node.output) > 1:
            return False

        for input_node in node.input:
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
        elif isinstance(dataset, de.ZipDataset):
            op_type = OpName.ZIP
        elif isinstance(dataset, de.MapDataset):
            op_type = OpName.MAP
        elif isinstance(dataset, de.RepeatDataset):
            op_type = OpName.REPEAT
        elif isinstance(dataset, de.StorageDataset):
            op_type = OpName.STORAGE
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
        elif isinstance(dataset, de.Cifar10Dataset):
            op_type = OpName.CIFAR10
        elif isinstance(dataset, de.Cifar100Dataset):
            op_type = OpName.CIFAR100
        elif isinstance(dataset, de.CelebADataset):
            op_type = OpName.CELEBA
        else:
            raise ValueError("Unsupported DatasetOp")

        return op_type

    # Convert python node into C node and add to C layer execution tree in postorder traversal.
    def __convert_node_postorder(self, node):
        op_type = self.__get_dataset_type(node)
        c_node = self.depipeline.AddNodeToTree(op_type, node.get_args())

        for py_child in node.input:
            c_child = self.__convert_node_postorder(py_child)
            self.depipeline.AddChildToParentNode(c_child, c_node)

        return c_node

    def __batch_node(self, dataset, level):
        """Recursively get batch node in the dataset tree."""
        if isinstance(dataset, de.BatchDataset):
            return
        for input_op in dataset.input:
            self.__batch_node(input_op, level + 1)

    @staticmethod
    def __print_local(dataset, level):
        """Recursively print the name and address of nodes in the dataset tree."""
        name = dataset.__class__.__name__
        ptr = hex(id(dataset))
        for _ in range(level):
            logger.info("\t", end='')
        if not dataset.input:
            logger.info("-%s (%s)", name, ptr)
        else:
            logger.info("+%s (%s)", name, ptr)
        for input_op in dataset.input:
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
