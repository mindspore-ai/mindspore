/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PARALLEL_AUTO_PARALLEL_REC_GENERATE_STRATEGY_H_
#define PARALLEL_AUTO_PARALLEL_REC_GENERATE_STRATEGY_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <list>

#include "frontend/parallel/auto_parallel/rec_core/rec_graph.h"
#include "frontend/parallel/ops_info/operator_info.h"

namespace mindspore {
namespace parallel {
void GenerateStrategy(const std::shared_ptr<Graph> &graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                      const std::shared_ptr<std::vector<std::vector<size_t>>> &eli_list,
                      const std::vector<std::vector<std::string>> &input_tensor_names,
                      const std::shared_ptr<std::vector<size_t>> &index_list, bool is_training,
                      const std::vector<std::vector<size_t>> &shared_tensors_ops, const FuncGraphPtr &root);
Dimensions PrepareMatMulStrategy(const std::shared_ptr<Graph> &graph, const size_t iter_graph, bool transpose_a,
                                 bool transpose_b, size_t iter_op_inputs);
Strategies PrepareMatMul(const std::shared_ptr<Graph> &graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                         const size_t iter_graph, const size_t iter_ops);
Strategies PrepareBiasAdd(const std::shared_ptr<Dimensions> &s);
Strategies PrepareStridedSlice(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                               Dimensions basic_stra);
Strategies PrepareSoftMax(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                          const Dimensions &basic_stra);
Strategies PrepareOneHot(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops, Dimensions s);
Strategies PrepareAxisRelatedStrategy(const std::shared_ptr<Graph> &graph,
                                      const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_graph,
                                      const size_t iter_ops);
Strategies PrepareGatherV2(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops, Dimensions s);
Dimensions PrepareGatherV2OutputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                         const size_t incoming_op_index);
Strategies PrepareL2Normalize(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                              Dimensions s);
Strategies MakeRecSearchStrategy(const std::shared_ptr<Graph> &graph,
                                 const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_graph,
                                 const size_t iter_ops);
Strategies CheckBroadcast(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                          const Dimensions &s);
Dimensions ApplyBroadcast(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops, Dimensions s,
                          size_t first_tensor_dim, size_t second_tensor_dim, bool broadcast_first_tensor);
Strategies CheckDivisible(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                          const Dimensions &s);
Strategies MakeDataParallelStrategy(const std::shared_ptr<Graph> &graph,
                                    const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_graph,
                                    const size_t iter_ops);
Strategies MakeFullBatchStrategy(const std::shared_ptr<Graph> &graph,
                                 const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_graph,
                                 const size_t iter_ops);
void SetBackToRawStrategy(const std::shared_ptr<OperatorInfo> &op);
Strategies PrepareStrategy(const std::shared_ptr<Graph> &graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                           const size_t iter_graph, const size_t iter_ops);
void GeneratePartitionedOperatorStrategy(const std::shared_ptr<Graph> &graph,
                                         const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                         const std::shared_ptr<std::vector<size_t>> &index_list);
size_t FindIndexOfOperatorIncoming(const std::vector<std::vector<std::string>> &input_tensor_names,
                                   const size_t iter_ops);
Dimensions CopyIncomingOperatorOutputStrategy(const std::shared_ptr<Graph> &graph,
                                              const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                              const size_t iter_ops, const size_t iter_graph,
                                              const size_t incoming_op_index);
Dimensions PrepareReshapeOutputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                        const size_t incoming_op_index);
Dimensions PrepareTransposeOutputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                          const size_t incoming_op_index);
Dimensions PrepareExpandDimsOutputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                           const size_t incoming_op_index);
Dimensions PrepareIncompingArithmeticOpeartorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                           const size_t incoming_op_index);
Dimensions PrepareIncomingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                const size_t incoming_op_index);
Dimensions GetAxisList(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const int64_t iter_ops);
Dimensions ModifyStrategyIfSqueezeIncoming(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                           const size_t incoming_op_index, Dimensions s);
bool GetKeepDims(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops);
Dimensions GetDimList(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops);
Dimensions ModifyStrategyIfReduceIncoming(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                          const size_t incoming_op_index, Dimensions s);
Dimensions GetDimListFromAttrs(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops);
Dimensions ModifyStrategyIfArgIncoming(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                       const size_t incoming_op_index, Dimensions s);
Dimensions CopyIncomingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                             const size_t iter_ops, const size_t incoming_op_index);
Strategies GenerateStrategiesFromStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                                          Dimensions basic_stra);
void GenerateEliminatedOperatorStrategyForward(const std::shared_ptr<Graph> &graph,
                                               const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                               const std::vector<std::vector<std::string>> &input_tensor_names,
                                               const std::shared_ptr<std::vector<size_t>> &index_list,
                                               const std::shared_ptr<std::vector<size_t>> &no_stra_op_list);
Dimensions ModifyStrategyIfSqueezeOutgoing(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                                           Dimensions s);
Dimensions CopyOutgoingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                             const std::vector<std::vector<std::string>> &input_tensor_names,
                                             const size_t iter_ops);
void GenerateEliminatedOperatorStrategyBackward(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                const std::vector<std::vector<std::string>> &input_tensor_names,
                                                const std::shared_ptr<std::vector<size_t>> &no_stra_op_list);
void GenerateRemainingOperatorStrategy(const std::shared_ptr<Graph> &graph,
                                       const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                       const std::vector<std::vector<std::string>> &input_tensor_names,
                                       const std::shared_ptr<std::vector<size_t>> &index_list,
                                       const std::shared_ptr<std::vector<size_t>> &no_stra_op_list);
void ModifyParamSharingOpsStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                   const std::vector<std::vector<size_t>> &param_users_ops_index);
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_REC_GENERATE_STRATEGY_H_
