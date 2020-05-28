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

#include "parallel/auto_parallel/rec_core/rec_graph.h"
#include "parallel/ops_info/operator_info.h"

namespace mindspore {
namespace parallel {
void GenerateStrategy(std::shared_ptr<Graph> graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                      const std::shared_ptr<std::vector<std::vector<size_t>>> eli_list,
                      const std::vector<std::vector<std::string>> &input_tensor_names,
                      const std::shared_ptr<std::vector<size_t>> index_list);
std::vector<std::vector<int32_t>> PrepareMatMul(const std::shared_ptr<Graph> &graph,
                                                const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                const size_t iter_graph, const size_t iter_ops);
std::vector<std::vector<int32_t>> PrepareVirtualDataset(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                        const size_t iter_ops);
std::vector<std::vector<int32_t>> PrepareBiasAdd(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                 const size_t iter_ops, std::vector<int32_t> s);
std::vector<std::vector<int32_t>> PrepareOneHot(std::vector<int32_t> s);
std::vector<std::vector<int32_t>> MakeRecSearchStrategy(const std::shared_ptr<Graph> &graph,
                                                        const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                        const size_t iter_graph, const size_t iter_ops);
std::vector<std::vector<int32_t>> MakeDataParallelStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                           const size_t iter_ops);
std::vector<std::vector<int32_t>> PrepareStrategy(const std::shared_ptr<Graph> &graph,
                                                  const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                  const size_t iter_graph, const size_t iter_ops);
void GeneratePartitionedOperatorStrategy(const std::shared_ptr<Graph> graph,
                                         const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                         const std::shared_ptr<std::vector<size_t>> index_list);
int FindIndexOfOperatorIncoming(const std::vector<std::vector<std::string>> &input_tensor_names, const size_t iter_ops);
std::vector<int32_t> CopyIncomingOperatorOutputStrategy(const std::shared_ptr<Graph> graph,
                                                        const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                        const size_t iter_ops, const size_t iter_graph);
std::vector<int32_t> PrepareIncomingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                          const int incoming_op_index);
std::vector<int32_t> GetAxisList(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const int iter_ops);
std::vector<int32_t> ModifyStrategyIfSqueezeIncoming(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                     const int incoming_op_index, std::vector<int32_t> s);
std::vector<int32_t> GetDimList(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops);
std::vector<int32_t> ModifyStrategyIfReduceIncoming(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                    const int incoming_op_index, std::vector<int32_t> s);
std::vector<int32_t> CopyIncomingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                       const int incoming_op_index, const size_t iter_ops,
                                                       const std::shared_ptr<std::vector<size_t>> no_stra_op_list);
std::vector<std::vector<int32_t>> GenerateStrategiesFromStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                                 const size_t iter_ops, std::vector<int32_t> s);
void GenerateEliminatedOperatorStrategyForward(std::shared_ptr<Graph> graph,
                                               const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                               const std::shared_ptr<std::vector<std::vector<size_t>>> eli_list,
                                               const std::vector<std::vector<std::string>> &input_tensor_names,
                                               const std::shared_ptr<std::vector<size_t>> index_list,
                                               const std::shared_ptr<std::vector<size_t>> no_stra_op_list);
std::vector<int32_t> ModifyStrategyIfSqueezeOutgoing(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                     const size_t iter_ops, std::vector<int32_t> s);
std::vector<int32_t> ModifyStrategyIfReduceOutgoing(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                    const size_t iter_ops, std::vector<int32_t> s);
std::vector<int32_t> CopyOutgoingOperatorInputStrategy(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                       const std::vector<std::vector<std::string>> &input_tensor_names,
                                                       const size_t iter_ops);
void GenerateEliminatedOperatorStrategyBackward(const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                                                const std::vector<std::vector<std::string>> &input_tensor_names,
                                                const std::shared_ptr<std::vector<size_t>> no_stra_op_list);
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_REC_GENERATE_STRATEGY_H_
