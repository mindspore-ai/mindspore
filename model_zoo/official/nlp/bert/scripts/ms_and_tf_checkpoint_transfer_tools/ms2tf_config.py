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

"""
    # tf and ms bert large checkpoint param transfer table
    # key:   tf
    # value: ms
"""
param_name_dict = {
    'bert/embeddings/word_embeddings': 'bert.bert.bert_embedding_lookup.embedding_table',
    'bert/embeddings/token_type_embeddings': 'bert.bert.bert_embedding_postprocessor.embedding_table',
    'bert/embeddings/position_embeddings': 'bert.bert.bert_embedding_postprocessor.full_position_embeddings',
    'bert/embeddings/LayerNorm/gamma': 'bert.bert.bert_embedding_postprocessor.layernorm.gamma',
    'bert/embeddings/LayerNorm/beta': 'bert.bert.bert_embedding_postprocessor.layernorm.beta',
    'bert/encoder/layer_0/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.0.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_0/attention/self/query/bias': 'bert.bert.bert_encoder.layers.0.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_0/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.0.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_0/attention/self/key/bias': 'bert.bert.bert_encoder.layers.0.attention.attention'
                                                    '.key_layer.bias',
    'bert/encoder/layer_0/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.0.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_0/attention/self/value/bias': 'bert.bert.bert_encoder.layers.0.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_0/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.0.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_0/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.0.attention.output.dense.bias',
    'bert/encoder/layer_0/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.0.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_0/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.0.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_0/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.0.intermediate.weight',
    'bert/encoder/layer_0/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.0.intermediate.bias',
    'bert/encoder/layer_0/output/dense/kernel': 'bert.bert.bert_encoder.layers.0.output.dense.weight',
    'bert/encoder/layer_0/output/dense/bias': 'bert.bert.bert_encoder.layers.0.output.dense.bias',
    'bert/encoder/layer_0/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.0.output.layernorm.gamma',
    'bert/encoder/layer_0/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.0.output.layernorm.beta',
    'bert/encoder/layer_1/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.1.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_1/attention/self/query/bias': 'bert.bert.bert_encoder.layers.1.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_1/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.1.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_1/attention/self/key/bias': 'bert.bert.bert_encoder.layers.1.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_1/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.1.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_1/attention/self/value/bias': 'bert.bert.bert_encoder.layers.1.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_1/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.1.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_1/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.1.attention.output.dense.bias',
    'bert/encoder/layer_1/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.1.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_1/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.1.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_1/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.1.intermediate.weight',
    'bert/encoder/layer_1/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.1.intermediate.bias',
    'bert/encoder/layer_1/output/dense/kernel': 'bert.bert.bert_encoder.layers.1.output.dense.weight',
    'bert/encoder/layer_1/output/dense/bias': 'bert.bert.bert_encoder.layers.1.output.dense.bias',
    'bert/encoder/layer_1/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.1.output.layernorm.gamma',
    'bert/encoder/layer_1/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.1.output.layernorm.beta',
    'bert/encoder/layer_2/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.2.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_2/attention/self/query/bias': 'bert.bert.bert_encoder.layers.2.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_2/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.2.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_2/attention/self/key/bias': 'bert.bert.bert_encoder.layers.2.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_2/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.2.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_2/attention/self/value/bias': 'bert.bert.bert_encoder.layers.2.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_2/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.2.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_2/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.2.attention.output.dense.bias',
    'bert/encoder/layer_2/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.2.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_2/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.2.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_2/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.2.intermediate.weight',
    'bert/encoder/layer_2/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.2.intermediate.bias',
    'bert/encoder/layer_2/output/dense/kernel': 'bert.bert.bert_encoder.layers.2.output.dense.weight',
    'bert/encoder/layer_2/output/dense/bias': 'bert.bert.bert_encoder.layers.2.output.dense.bias',
    'bert/encoder/layer_2/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.2.output.layernorm.gamma',
    'bert/encoder/layer_2/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.2.output.layernorm.beta',
    'bert/encoder/layer_3/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.3.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_3/attention/self/query/bias': 'bert.bert.bert_encoder.layers.3.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_3/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.3.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_3/attention/self/key/bias': 'bert.bert.bert_encoder.layers.3.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_3/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.3.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_3/attention/self/value/bias': 'bert.bert.bert_encoder.layers.3.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_3/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.3.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_3/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.3.attention.output.dense.bias',
    'bert/encoder/layer_3/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.3.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_3/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.3.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_3/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.3.intermediate.weight',
    'bert/encoder/layer_3/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.3.intermediate.bias',
    'bert/encoder/layer_3/output/dense/kernel': 'bert.bert.bert_encoder.layers.3.output.dense.weight',
    'bert/encoder/layer_3/output/dense/bias': 'bert.bert.bert_encoder.layers.3.output.dense.bias',
    'bert/encoder/layer_3/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.3.output.layernorm.gamma',
    'bert/encoder/layer_3/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.3.output.layernorm.beta',
    'bert/encoder/layer_4/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.4.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_4/attention/self/query/bias': 'bert.bert.bert_encoder.layers.4.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_4/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.4.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_4/attention/self/key/bias': 'bert.bert.bert_encoder.layers.4'
                                                    '.attention.attention.key_layer.bias',
    'bert/encoder/layer_4/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.4.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_4/attention/self/value/bias': 'bert.bert.bert_encoder.layers.4.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_4/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.4.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_4/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.4.attention.output.dense.bias',
    'bert/encoder/layer_4/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.4.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_4/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.4.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_4/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.4.intermediate.weight',
    'bert/encoder/layer_4/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.4.intermediate.bias',
    'bert/encoder/layer_4/output/dense/kernel': 'bert.bert.bert_encoder.layers.4.output.dense.weight',
    'bert/encoder/layer_4/output/dense/bias': 'bert.bert.bert_encoder.layers.4.output.dense.bias',
    'bert/encoder/layer_4/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.4.output.layernorm.gamma',
    'bert/encoder/layer_4/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.4.output.layernorm.beta',
    'bert/encoder/layer_5/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.5.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_5/attention/self/query/bias': 'bert.bert.bert_encoder.layers.5.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_5/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.5.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_5/attention/self/key/bias': 'bert.bert.bert_encoder.layers.5.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_5/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.5.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_5/attention/self/value/bias': 'bert.bert.bert_encoder.layers.5.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_5/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.5.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_5/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.5.attention.output.dense.bias',
    'bert/encoder/layer_5/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.5.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_5/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.5.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_5/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.5.intermediate.weight',
    'bert/encoder/layer_5/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.5.intermediate.bias',
    'bert/encoder/layer_5/output/dense/kernel': 'bert.bert.bert_encoder.layers.5.output.dense.weight',
    'bert/encoder/layer_5/output/dense/bias': 'bert.bert.bert_encoder.layers.5.output.dense.bias',
    'bert/encoder/layer_5/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.5.output.layernorm.gamma',
    'bert/encoder/layer_5/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.5.output.layernorm.beta',
    'bert/encoder/layer_6/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.6.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_6/attention/self/query/bias': 'bert.bert.bert_encoder.layers.6.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_6/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.6.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_6/attention/self/key/bias': 'bert.bert.bert_encoder.layers.6.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_6/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.6.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_6/attention/self/value/bias': 'bert.bert.bert_encoder.layers.6.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_6/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.6.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_6/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.6.attention.output.dense.bias',
    'bert/encoder/layer_6/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.6.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_6/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.6.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_6/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.6.intermediate.weight',
    'bert/encoder/layer_6/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.6.intermediate.bias',
    'bert/encoder/layer_6/output/dense/kernel': 'bert.bert.bert_encoder.layers.6.output.dense.weight',
    'bert/encoder/layer_6/output/dense/bias': 'bert.bert.bert_encoder.layers.6.output.dense.bias',
    'bert/encoder/layer_6/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.6.output.layernorm.gamma',
    'bert/encoder/layer_6/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.6.output.layernorm.beta',
    'bert/encoder/layer_7/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.7.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_7/attention/self/query/bias': 'bert.bert.bert_encoder.layers.7.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_7/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.7.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_7/attention/self/key/bias': 'bert.bert.bert_encoder.layers.7.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_7/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.7.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_7/attention/self/value/bias': 'bert.bert.bert_encoder.layers.7.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_7/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.7.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_7/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.7.attention.output.dense.bias',
    'bert/encoder/layer_7/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.7.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_7/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.7.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_7/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.7.intermediate.weight',
    'bert/encoder/layer_7/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.7.intermediate.bias',
    'bert/encoder/layer_7/output/dense/kernel': 'bert.bert.bert_encoder.layers.7.output.dense.weight',
    'bert/encoder/layer_7/output/dense/bias': 'bert.bert.bert_encoder.layers.7.output.dense.bias',
    'bert/encoder/layer_7/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.7.output.layernorm.gamma',
    'bert/encoder/layer_7/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.7.output.layernorm.beta',
    'bert/encoder/layer_8/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.8.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_8/attention/self/query/bias': 'bert.bert.bert_encoder.layers.8.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_8/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.8.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_8/attention/self/key/bias': 'bert.bert.bert_encoder.layers.8.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_8/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.8.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_8/attention/self/value/bias': 'bert.bert.bert_encoder.layers.8.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_8/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.8.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_8/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.8.attention.output.dense.bias',
    'bert/encoder/layer_8/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.8.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_8/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.8.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_8/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.8.intermediate.weight',
    'bert/encoder/layer_8/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.8.intermediate.bias',
    'bert/encoder/layer_8/output/dense/kernel': 'bert.bert.bert_encoder.layers.8.output.dense.weight',
    'bert/encoder/layer_8/output/dense/bias': 'bert.bert.bert_encoder.layers.8.output.dense.bias',
    'bert/encoder/layer_8/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.8.output.layernorm.gamma',
    'bert/encoder/layer_8/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.8.output.layernorm.beta',
    'bert/encoder/layer_9/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.9.attention.attention'
                                                        '.query_layer.weight',
    'bert/encoder/layer_9/attention/self/query/bias': 'bert.bert.bert_encoder.layers.9.attention.attention'
                                                      '.query_layer.bias',
    'bert/encoder/layer_9/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.9.attention.attention.key_layer'
                                                      '.weight',
    'bert/encoder/layer_9/attention/self/key/bias': 'bert.bert.bert_encoder.layers.9.attention'
                                                    '.attention.key_layer.bias',
    'bert/encoder/layer_9/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.9.attention.attention'
                                                        '.value_layer.weight',
    'bert/encoder/layer_9/attention/self/value/bias': 'bert.bert.bert_encoder.layers.9.attention.attention'
                                                      '.value_layer.bias',
    'bert/encoder/layer_9/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.9.attention.output.dense'
                                                          '.weight',
    'bert/encoder/layer_9/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.9.attention.output.dense.bias',
    'bert/encoder/layer_9/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.9.attention.output'
                                                             '.layernorm.gamma',
    'bert/encoder/layer_9/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.9.attention.output'
                                                            '.layernorm.beta',
    'bert/encoder/layer_9/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.9.intermediate.weight',
    'bert/encoder/layer_9/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.9.intermediate.bias',
    'bert/encoder/layer_9/output/dense/kernel': 'bert.bert.bert_encoder.layers.9.output.dense.weight',
    'bert/encoder/layer_9/output/dense/bias': 'bert.bert.bert_encoder.layers.9.output.dense.bias',
    'bert/encoder/layer_9/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.9.output.layernorm.gamma',
    'bert/encoder/layer_9/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.9.output.layernorm.beta',
    'bert/encoder/layer_10/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.10.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_10/attention/self/query/bias': 'bert.bert.bert_encoder.layers.10.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_10/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.10.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_10/attention/self/key/bias': 'bert.bert.bert_encoder.layers.10.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_10/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.10.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_10/attention/self/value/bias': 'bert.bert.bert_encoder.layers.10.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_10/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.10.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_10/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.10.attention.output.dense.bias',
    'bert/encoder/layer_10/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.10.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_10/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.10.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_10/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.10.intermediate.weight',
    'bert/encoder/layer_10/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.10.intermediate.bias',
    'bert/encoder/layer_10/output/dense/kernel': 'bert.bert.bert_encoder.layers.10.output.dense.weight',
    'bert/encoder/layer_10/output/dense/bias': 'bert.bert.bert_encoder.layers.10.output.dense.bias',
    'bert/encoder/layer_10/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.10.output.layernorm.gamma',
    'bert/encoder/layer_10/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.10.output.layernorm.beta',
    'bert/encoder/layer_11/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.11.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_11/attention/self/query/bias': 'bert.bert.bert_encoder.layers.11.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_11/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.11.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_11/attention/self/key/bias': 'bert.bert.bert_encoder.layers.11.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_11/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.11.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_11/attention/self/value/bias': 'bert.bert.bert_encoder.layers.11.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_11/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.11.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_11/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.11.attention.output.dense.bias',
    'bert/encoder/layer_11/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.11.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_11/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.11.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_11/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.11.intermediate.weight',
    'bert/encoder/layer_11/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.11.intermediate.bias',
    'bert/encoder/layer_11/output/dense/kernel': 'bert.bert.bert_encoder.layers.11.output.dense.weight',
    'bert/encoder/layer_11/output/dense/bias': 'bert.bert.bert_encoder.layers.11.output.dense.bias',
    'bert/encoder/layer_11/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.11.output.layernorm.gamma',
    'bert/encoder/layer_11/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.11.output.layernorm.beta',
    'bert/encoder/layer_12/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.12.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_12/attention/self/query/bias': 'bert.bert.bert_encoder.layers.12.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_12/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.12.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_12/attention/self/key/bias': 'bert.bert.bert_encoder.layers.12.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_12/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.12.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_12/attention/self/value/bias': 'bert.bert.bert_encoder.layers.12.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_12/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.12.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_12/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.12.attention.output.dense.bias',
    'bert/encoder/layer_12/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.12.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_12/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.12.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_12/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.12.intermediate.weight',
    'bert/encoder/layer_12/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.12.intermediate.bias',
    'bert/encoder/layer_12/output/dense/kernel': 'bert.bert.bert_encoder.layers.12.output.dense.weight',
    'bert/encoder/layer_12/output/dense/bias': 'bert.bert.bert_encoder.layers.12.output.dense.bias',
    'bert/encoder/layer_12/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.12.output.layernorm.gamma',
    'bert/encoder/layer_12/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.12.output.layernorm.beta',
    'bert/encoder/layer_13/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.13.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_13/attention/self/query/bias': 'bert.bert.bert_encoder.layers.13.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_13/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.13.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_13/attention/self/key/bias': 'bert.bert.bert_encoder.layers.13.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_13/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.13.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_13/attention/self/value/bias': 'bert.bert.bert_encoder.layers.13.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_13/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.13.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_13/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.13.attention.output.dense.bias',
    'bert/encoder/layer_13/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.13.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_13/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.13.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_13/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.13.intermediate.weight',
    'bert/encoder/layer_13/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.13.intermediate.bias',
    'bert/encoder/layer_13/output/dense/kernel': 'bert.bert.bert_encoder.layers.13.output.dense.weight',
    'bert/encoder/layer_13/output/dense/bias': 'bert.bert.bert_encoder.layers.13.output.dense.bias',
    'bert/encoder/layer_13/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.13.output.layernorm.gamma',
    'bert/encoder/layer_13/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.13.output.layernorm.beta',
    'bert/encoder/layer_14/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.14.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_14/attention/self/query/bias': 'bert.bert.bert_encoder.layers.14.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_14/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.14.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_14/attention/self/key/bias': 'bert.bert.bert_encoder.layers.14.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_14/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.14.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_14/attention/self/value/bias': 'bert.bert.bert_encoder.layers.14.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_14/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.14.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_14/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.14.attention.output.dense.bias',
    'bert/encoder/layer_14/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.14.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_14/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.14.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_14/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.14.intermediate.weight',
    'bert/encoder/layer_14/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.14.intermediate.bias',
    'bert/encoder/layer_14/output/dense/kernel': 'bert.bert.bert_encoder.layers.14.output.dense.weight',
    'bert/encoder/layer_14/output/dense/bias': 'bert.bert.bert_encoder.layers.14.output.dense.bias',
    'bert/encoder/layer_14/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.14.output.layernorm.gamma',
    'bert/encoder/layer_14/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.14.output.layernorm.beta',
    'bert/encoder/layer_15/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.15.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_15/attention/self/query/bias': 'bert.bert.bert_encoder.layers.15.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_15/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.15.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_15/attention/self/key/bias': 'bert.bert.bert_encoder.layers.15.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_15/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.15.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_15/attention/self/value/bias': 'bert.bert.bert_encoder.layers.15.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_15/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.15.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_15/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.15.attention.output.dense.bias',
    'bert/encoder/layer_15/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.15.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_15/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.15.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_15/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.15.intermediate.weight',
    'bert/encoder/layer_15/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.15.intermediate.bias',
    'bert/encoder/layer_15/output/dense/kernel': 'bert.bert.bert_encoder.layers.15.output.dense.weight',
    'bert/encoder/layer_15/output/dense/bias': 'bert.bert.bert_encoder.layers.15.output.dense.bias',
    'bert/encoder/layer_15/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.15.output.layernorm.gamma',
    'bert/encoder/layer_15/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.15.output.layernorm.beta',
    'bert/encoder/layer_16/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.16.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_16/attention/self/query/bias': 'bert.bert.bert_encoder.layers.16.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_16/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.16.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_16/attention/self/key/bias': 'bert.bert.bert_encoder.layers.16.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_16/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.16.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_16/attention/self/value/bias': 'bert.bert.bert_encoder.layers.16.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_16/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.16.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_16/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.16.attention.output.dense.bias',
    'bert/encoder/layer_16/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.16.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_16/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.16.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_16/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.16.intermediate.weight',
    'bert/encoder/layer_16/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.16.intermediate.bias',
    'bert/encoder/layer_16/output/dense/kernel': 'bert.bert.bert_encoder.layers.16.output.dense.weight',
    'bert/encoder/layer_16/output/dense/bias': 'bert.bert.bert_encoder.layers.16.output.dense.bias',
    'bert/encoder/layer_16/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.16.output.layernorm.gamma',
    'bert/encoder/layer_16/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.16.output.layernorm.beta',
    'bert/encoder/layer_17/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.17.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_17/attention/self/query/bias': 'bert.bert.bert_encoder.layers.17.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_17/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.17.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_17/attention/self/key/bias': 'bert.bert.bert_encoder.layers.17.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_17/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.17.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_17/attention/self/value/bias': 'bert.bert.bert_encoder.layers.17.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_17/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.17.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_17/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.17.attention.output.dense.bias',
    'bert/encoder/layer_17/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.17.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_17/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.17.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_17/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.17.intermediate.weight',
    'bert/encoder/layer_17/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.17.intermediate.bias',
    'bert/encoder/layer_17/output/dense/kernel': 'bert.bert.bert_encoder.layers.17.output.dense.weight',
    'bert/encoder/layer_17/output/dense/bias': 'bert.bert.bert_encoder.layers.17.output.dense.bias',
    'bert/encoder/layer_17/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.17.output.layernorm.gamma',
    'bert/encoder/layer_17/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.17.output.layernorm.beta',
    'bert/encoder/layer_18/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.18.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_18/attention/self/query/bias': 'bert.bert.bert_encoder.layers.18.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_18/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.18.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_18/attention/self/key/bias': 'bert.bert.bert_encoder.layers.18.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_18/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.18.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_18/attention/self/value/bias': 'bert.bert.bert_encoder.layers.18.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_18/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.18.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_18/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.18.attention.output.dense.bias',
    'bert/encoder/layer_18/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.18.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_18/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.18.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_18/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.18.intermediate.weight',
    'bert/encoder/layer_18/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.18.intermediate.bias',
    'bert/encoder/layer_18/output/dense/kernel': 'bert.bert.bert_encoder.layers.18.output.dense.weight',
    'bert/encoder/layer_18/output/dense/bias': 'bert.bert.bert_encoder.layers.18.output.dense.bias',
    'bert/encoder/layer_18/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.18.output.layernorm.gamma',
    'bert/encoder/layer_18/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.18.output.layernorm.beta',
    'bert/encoder/layer_19/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.19.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_19/attention/self/query/bias': 'bert.bert.bert_encoder.layers.19.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_19/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.19.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_19/attention/self/key/bias': 'bert.bert.bert_encoder.layers.19.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_19/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.19.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_19/attention/self/value/bias': 'bert.bert.bert_encoder.layers.19.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_19/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.19.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_19/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.19.attention.output.dense.bias',
    'bert/encoder/layer_19/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.19.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_19/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.19.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_19/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.19.intermediate.weight',
    'bert/encoder/layer_19/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.19.intermediate.bias',
    'bert/encoder/layer_19/output/dense/kernel': 'bert.bert.bert_encoder.layers.19.output.dense.weight',
    'bert/encoder/layer_19/output/dense/bias': 'bert.bert.bert_encoder.layers.19.output.dense.bias',
    'bert/encoder/layer_19/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.19.output.layernorm.gamma',
    'bert/encoder/layer_19/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.19.output.layernorm.beta',
    'bert/encoder/layer_20/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.20.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_20/attention/self/query/bias': 'bert.bert.bert_encoder.layers.20.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_20/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.20.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_20/attention/self/key/bias': 'bert.bert.bert_encoder.layers.20.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_20/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.20.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_20/attention/self/value/bias': 'bert.bert.bert_encoder.layers.20.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_20/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.20.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_20/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.20.attention.output.dense.bias',
    'bert/encoder/layer_20/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.20.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_20/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.20.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_20/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.20.intermediate.weight',
    'bert/encoder/layer_20/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.20.intermediate.bias',
    'bert/encoder/layer_20/output/dense/kernel': 'bert.bert.bert_encoder.layers.20.output.dense.weight',
    'bert/encoder/layer_20/output/dense/bias': 'bert.bert.bert_encoder.layers.20.output.dense.bias',
    'bert/encoder/layer_20/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.20.output.layernorm.gamma',
    'bert/encoder/layer_20/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.20.output.layernorm.beta',
    'bert/encoder/layer_21/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.21.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_21/attention/self/query/bias': 'bert.bert.bert_encoder.layers.21.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_21/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.21.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_21/attention/self/key/bias': 'bert.bert.bert_encoder.layers.21.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_21/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.21.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_21/attention/self/value/bias': 'bert.bert.bert_encoder.layers.21.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_21/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.21.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_21/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.21.attention.output.dense.bias',
    'bert/encoder/layer_21/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.21.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_21/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.21.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_21/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.21.intermediate.weight',
    'bert/encoder/layer_21/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.21.intermediate.bias',
    'bert/encoder/layer_21/output/dense/kernel': 'bert.bert.bert_encoder.layers.21.output.dense.weight',
    'bert/encoder/layer_21/output/dense/bias': 'bert.bert.bert_encoder.layers.21.output.dense.bias',
    'bert/encoder/layer_21/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.21.output.layernorm.gamma',
    'bert/encoder/layer_21/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.21.output.layernorm.beta',
    'bert/encoder/layer_22/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.22.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_22/attention/self/query/bias': 'bert.bert.bert_encoder.layers.22.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_22/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.22.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_22/attention/self/key/bias': 'bert.bert.bert_encoder.layers.22.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_22/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.22.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_22/attention/self/value/bias': 'bert.bert.bert_encoder.layers.22.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_22/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.22.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_22/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.22.attention.output.dense.bias',
    'bert/encoder/layer_22/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.22.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_22/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.22.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_22/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.22.intermediate.weight',
    'bert/encoder/layer_22/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.22.intermediate.bias',
    'bert/encoder/layer_22/output/dense/kernel': 'bert.bert.bert_encoder.layers.22.output.dense.weight',
    'bert/encoder/layer_22/output/dense/bias': 'bert.bert.bert_encoder.layers.22.output.dense.bias',
    'bert/encoder/layer_22/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.22.output.layernorm.gamma',
    'bert/encoder/layer_22/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.22.output.layernorm.beta',
    'bert/encoder/layer_23/attention/self/query/kernel': 'bert.bert.bert_encoder.layers.23.attention.attention'
                                                         '.query_layer.weight',
    'bert/encoder/layer_23/attention/self/query/bias': 'bert.bert.bert_encoder.layers.23.attention.attention'
                                                       '.query_layer.bias',
    'bert/encoder/layer_23/attention/self/key/kernel': 'bert.bert.bert_encoder.layers.23.attention.attention'
                                                       '.key_layer.weight',
    'bert/encoder/layer_23/attention/self/key/bias': 'bert.bert.bert_encoder.layers.23.attention.attention.key_layer'
                                                     '.bias',
    'bert/encoder/layer_23/attention/self/value/kernel': 'bert.bert.bert_encoder.layers.23.attention.attention'
                                                         '.value_layer.weight',
    'bert/encoder/layer_23/attention/self/value/bias': 'bert.bert.bert_encoder.layers.23.attention.attention'
                                                       '.value_layer.bias',
    'bert/encoder/layer_23/attention/output/dense/kernel': 'bert.bert.bert_encoder.layers.23.attention.output.dense'
                                                           '.weight',
    'bert/encoder/layer_23/attention/output/dense/bias': 'bert.bert.bert_encoder.layers.23.attention.output.dense.bias',
    'bert/encoder/layer_23/attention/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.23.attention.output'
                                                              '.layernorm.gamma',
    'bert/encoder/layer_23/attention/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.23.attention.output'
                                                             '.layernorm.beta',
    'bert/encoder/layer_23/intermediate/dense/kernel': 'bert.bert.bert_encoder.layers.23.intermediate.weight',
    'bert/encoder/layer_23/intermediate/dense/bias': 'bert.bert.bert_encoder.layers.23.intermediate.bias',
    'bert/encoder/layer_23/output/dense/kernel': 'bert.bert.bert_encoder.layers.23.output.dense.weight',
    'bert/encoder/layer_23/output/dense/bias': 'bert.bert.bert_encoder.layers.23.output.dense.bias',
    'bert/encoder/layer_23/output/LayerNorm/gamma': 'bert.bert.bert_encoder.layers.23.output.layernorm.gamma',
    'bert/encoder/layer_23/output/LayerNorm/beta': 'bert.bert.bert_encoder.layers.23.output.layernorm.beta',
    'bert/pooler/dense/kernel': 'bert.bert.dense.weight',
    'bert/pooler/dense/bias': 'bert.bert.dense.bias',
    'cls/predictions/output_bias': 'bert.cls1.output_bias',
    'cls/predictions/transform/dense/kernel': 'bert.cls1.dense.weight',
    'cls/predictions/transform/dense/bias': 'bert.cls1.dense.bias',
    'cls/predictions/transform/LayerNorm/gamma': 'bert.cls1.layernorm.gamma',
    'cls/predictions/transform/LayerNorm/beta': 'bert.cls1.layernorm.beta',
    'cls/seq_relationship/output_weights': 'bert.cls2.dense.weight',
    'cls/seq_relationship/output_bias': 'bert.cls2.dense.bias',
}
