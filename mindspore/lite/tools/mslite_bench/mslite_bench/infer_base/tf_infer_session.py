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
        super(TFSession, self).__init__(model_file, cfg)
        self.graph = None
        self.model_session = self._create_infer_session()

        self.input_tensor_map = {
            tensor_name: self.graph.get_tensor_by_name(tensor_name + ': 0') for
            tensor_name in self.input_tensor_shapes.keys()
        }
        self.logger.info(f'[TF INFER]:input tensor is {self.input_tensor_map}')

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
        with tf.gfile.GFile(self.model_file, 'rb') as f:
            self.logger.info(f'tf_session_create: Read {self.model_file} successfully')
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            input_tensor_map = self._get_tf_input_tensor_map(graph_def)
            self.logger.info(f'[TF INFER] input tensor map is {input_tensor_map}')
            tf.import_graph_def(graph_def, input_map=input_tensor_map, name='')
            self.logger.info('Tensor map done')
            self.graph = tf.get_default_graph()
        model_session = tf.Session(graph=self.graph)
        return model_session

    def _get_tf_input_tensor_map(self, graph_def):
        """get tensorflow input tensor map"""
        input_tensor_map = {}
        tf.import_graph_def(graph_def, name='')
        default_graph = tf.get_default_graph()
        for key, shape in self.input_tensor_shapes.items():
            tensor_name = f'{key}:0'
            input_tensor = default_graph.get_tensor_by_name(tensor_name)
            self.logger.info(f'input tensor info {input_tensor}')
            input_tensor.set_shape(shape)
            input_tensor_map[key] = input_tensor
        return input_tensor_map
