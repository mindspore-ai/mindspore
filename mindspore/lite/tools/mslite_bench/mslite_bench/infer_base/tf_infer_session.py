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
"""
for paddle infer session
"""
import os
from abc import ABC
from typing import Dict

import tensorflow as tf
import numpy as np

from mslite_bench.infer_base.abs_infer_session import AbcInferSession


class TFSession(AbcInferSession, ABC):
    """TF infer session"""
    def __init__(self,
                 model_file,
                 cfg=None):
        super().__init__(model_file, cfg)
        self.graph = None
        self.model_session = self._create_infer_session()

        self.input_tensor_map = {
            tensor_name: self.graph.get_tensor_by_name(tensor_name + ': 0') for
            tensor_name in self.input_tensor_shapes.keys()
        }

        self.output_tensor_map = {
            tensor_name: self.graph.get_tensor_by_name(tensor_name + ': 0') for
            tensor_name in self.output_tensor_names
        }

    def infer(self, input_data_map: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """model infer"""
        results = {
            key: self.model_session.run(output_tensor,
                                        feed_dict={
                                            self.input_tensor_map.get(name): input_data_map.get(name)
                                            for name in self.input_tensor_shapes.keys()
                                        })
            for key, output_tensor in self.output_tensor_map.items()
        }

        return results

    def _create_infer_session(self):
        """create infer session"""
        if not os.path.exists(self.model_file):
            raise ValueError(f'TF model {self.model_file} does not exist')
        with tf.io.gfile.GFile(self.model_file, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            input_tensor_map = self._get_tf_input_tensor_map(graph_def)
            tf.import_graph_def(graph_def, input_map=input_tensor_map, name='')
            self.logger.debug('Tensor map done')
            self.graph = tf.compat.v1.get_default_graph()
        model_session = tf.compat.v1.Session(graph=self.graph)
        return model_session

    def _get_tf_input_tensor_map(self, graph_def):
        """get tensorflow input tensor map"""
        input_tensor_map = {}
        tf.import_graph_def(graph_def, name='')
        default_graph = tf.compat.v1.get_default_graph()
        for key, shape in self.input_tensor_shapes.items():
            tensor_name = f'{key}:0'
            input_tensor = default_graph.get_tensor_by_name(tensor_name)
            input_tensor.set_shape(shape)
            input_tensor_map[key] = input_tensor
        return input_tensor_map
