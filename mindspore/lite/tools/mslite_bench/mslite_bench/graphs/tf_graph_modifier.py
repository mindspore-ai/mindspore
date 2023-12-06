# Copyright 2023 Huawei Technologies Co., Ltd
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
""" tensorflow graph modifier"""
import os
from abc import ABC

import tensorflow as tf

from mslite_bench.graphs.graph_modifier import ABCGraphModifier


class TFModifier(ABCGraphModifier, ABC):
    """ modifier for onnx model"""
    def __init__(self, model_path):
        super().__init__()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'{model_path} does not exist!')
        self.model_path = model_path
        self.graph_def = self._get_tf_graph()
        self.model_input_names = self._get_input_names()

    def extract_model(self,
                      save_path,
                      input_names=None,
                      output_names=None):
        """ extract sub model based on input and output tensor names"""
        if input_names is None:
            input_names = self.model_input_names

        if not isinstance(input_names, list) or not isinstance(output_names, list):
            raise ValueError("input and output nodes name shall be a list")


        sub_graph_def = tf.compat.v1.graph_util.extract_sub_graph(self.graph_def,
                                                                  output_names)

        with tf.io.gfile.GFile(save_path, 'wb') as f:
            f.write(sub_graph_def.SerializeToString())
        return output_names

    def _all_node_names(self):
        """return all node names in network"""
        raise NotImplementedError


    def _sorted_blocks(self):
        """get sorted blocks based on feed-foward network"""
        return []

    def _get_input_names(self):
        """get all input nodes"""
        return [node.name for node in self.graph_def.node if node.op == 'Placeholder']

    def _get_tf_graph(self):
        """get tensorflow graph"""
        graph_def = None
        with tf.io.gfile.GFile(self.model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def
