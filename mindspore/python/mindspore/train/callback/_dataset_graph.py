# Copyright 2020 Huawei Technologies Co., Ltd
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
# ============================================================================
"""Define dataset graph related operations."""
from __future__ import absolute_import

import json
from importlib import import_module

from mindspore import log as logger
from mindspore.train import lineage_pb2


class DatasetGraph:
    """Handle the data graph and packages it into binary data."""
    def package_dataset_graph(self, dataset):
        """
        packages dataset graph into binary data

        Args:
            dataset (MindDataset): Refer to MindDataset.

        Returns:
            DatasetGraph, a object of lineage_pb2.DatasetGraph.
        """
        dataset_package = import_module('mindspore.dataset')
        try:
            dataset_dict = dataset_package.serialize(dataset)
        except (TypeError, OSError) as exc:
            logger.warning("Summary can not collect dataset graph, there is an error in dataset internal, "
                           "detail: %s.", str(exc))
            return None

        dataset_graph_proto = lineage_pb2.DatasetGraph()
        if not isinstance(dataset_dict, dict):
            logger.warning("The dataset graph serialized from dataset object is not a dict. "
                           "Its type is %r.", type(dataset_dict).__name__)
            return dataset_graph_proto
        if "children" in dataset_dict:
            children = dataset_dict.pop("children")
            if children:
                self._package_children(children=children, message=dataset_graph_proto)
            self._package_current_dataset(operation=dataset_dict, message=dataset_graph_proto)
        return dataset_graph_proto

    def _package_children(self, children, message):
        """
        Package children in dataset operation.

        Args:
            children (list[dict]): Child operations.
            message (DatasetGraph): Children proto message.
        """
        for child in children:
            if child:
                child_graph_message = getattr(message, "children").add()
                grandson = child.pop("children")
                if grandson:
                    self._package_children(children=grandson, message=child_graph_message)
                # package other parameters
                self._package_current_dataset(operation=child, message=child_graph_message)

    def _package_current_dataset(self, operation, message):
        """
        Package operation parameters in event message.

        Args:
            operation (dict): Operation dict.
            message (Operation): Operation proto message.
        """
        for key, value in operation.items():
            if value and key == "operations":
                for operator in value:
                    self._package_enhancement_operation(
                        operator,
                        message.operations.add()
                    )
            elif value and key == "sampler":
                self._package_enhancement_operation(
                    value,
                    message.sampler
                )
            else:
                self._package_parameter(key, value, message.parameter)

    def _package_enhancement_operation(self, operation, message):
        """
        Package enhancement operation in MapDataset.

        Args:
            operation (dict): Enhancement operation.
            message (Operation): Enhancement operation proto message.
        """
        if operation is None:
            logger.warning("Summary cannot collect the operation for dataset graph as the operation is none."
                           "it may due to the custom operation cannot be pickled.")
            return
        for key, value in operation.items():
            if isinstance(value, (list, tuple)):
                if all(isinstance(ele, int) for ele in value):
                    message.size.extend(value)
                else:
                    message.weights.extend(value)
            else:
                self._package_parameter(key, value, message.operationParam)

    @staticmethod
    def _package_parameter(key, value, message):
        """
        Package parameters in operation.

        Args:
            key (str): Operation name.
            value (Union[str, bool, int, float, list, None]): Operation args.
            message (OperationParameter): Operation proto message.
        """
        if isinstance(value, str):
            message.mapStr[key] = value
        elif isinstance(value, bool):
            message.mapBool[key] = value
        elif isinstance(value, int):
            message.mapInt[key] = value
        elif isinstance(value, float):
            message.mapDouble[key] = value
        elif isinstance(value, (list, tuple)) and key != "operations":
            if value:
                replace_value_list = list(map(lambda x: "" if x is None else json.dumps(x), value))
                message.mapStrList[key].strValue.extend(replace_value_list)
        elif isinstance(value, dict):
            try:
                message.mapStr[key] = json.dumps(value)
            except TypeError as exo:
                logger.warning("Transform the value of parameter %r to string failed. Detail: %s.", key, str(exo))
        elif value is None:
            message.mapStr[key] = "None"
        else:
            logger.warning("The parameter %r is not recorded, because its type is not supported in event package, "
                           "Its type should be in ['str', 'bool', 'int', 'float', '(list, tuple)', 'dict', 'None'], "
                           "but got type is %r.", key, type(value).__name__)
