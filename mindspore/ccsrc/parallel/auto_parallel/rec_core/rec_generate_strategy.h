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
void GenerateStrategy(const std::shared_ptr<Graph> graph, std::vector<std::shared_ptr<OperatorInfo>> ops,
                      const std::shared_ptr<std::vector<size_t>> ops_nodes_list,
                      const std::shared_ptr<std::vector<size_t>> index_list,
                      const std::shared_ptr<std::vector<std::vector<size_t>>> eli_list);
void PrepareMatMul(const std::shared_ptr<Graph> graph, const std::vector<std::shared_ptr<OperatorInfo>> &ops,
                   const size_t iter_ops, const size_t iter_nodes, const size_t iter_op_inputs, std::vector<int32_t> s);
void PrepareConv2D(const std::shared_ptr<Graph> graph, const size_t iter_nodes, const size_t iter_op_inputs,
                   std::vector<int32_t> s);
void PrepareBiasAdd(const std::shared_ptr<Graph> graph, const size_t iter_nodes, const size_t iter_op_inputs,
                    std::vector<int32_t> s);
void PrepareBN(const std::shared_ptr<Graph> graph, const size_t iter_nodes, const size_t iter_op_inputs,
               std::vector<int32_t> s);
void PrepareSparse(const size_t iter_op_inputs, std::vector<int32_t> s);
void RefillOrigin(const std::vector<std::shared_ptr<OperatorInfo>> &ops, const size_t iter_ops,
                  const size_t iter_op_inputs, std::vector<int32_t> s);
std::vector<int32_t> PrepareStrategy(const std::shared_ptr<Graph> graph,
                                     const std::vector<std::shared_ptr<OperatorInfo>> &ops, const std::string &type,
                                     const size_t iter_ops, const size_t iter_nodes, const size_t iter_op_inputs);
size_t IterNodes(const std::shared_ptr<std::vector<size_t>> ops_nodes_list,
                 const std::shared_ptr<std::vector<size_t>> index_list,
                 const std::shared_ptr<std::vector<std::vector<size_t>>> eli_list, const size_t iter_ops,
                 size_t iter_nodes);
void MaskNoSupportedOps(const std::shared_ptr<Graph> graph);
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_AUTO_PARALLEL_REC_GENERATE_STRATEGY_H_
