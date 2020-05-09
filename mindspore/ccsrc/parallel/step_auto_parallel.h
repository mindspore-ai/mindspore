/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef PARALLEL_STEP_AUTO_PARALLEL_H_
#define PARALLEL_STEP_AUTO_PARALLEL_H_

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "ir/anf.h"
#include "optimizer/opt.h"
#include "parallel/status.h"
#include "pipeline/pipeline.h"

namespace mindspore {
namespace parallel {
bool IsSplittableOperator(const std::string &);

bool IsAutoParallelCareNode(const CNodePtr &);

// main step of Auto-parallel
bool StepAutoParallel(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer);

size_t GetLengthOfDataType(const TypePtr &type);

std::vector<bool> ExtractInputParameterByNode(const CNodePtr &node);

std::vector<size_t> ExtractInputTypeLengthByNode(const CNodePtr &node);

std::vector<TypePtr> ExtractOutputTypeByNode(const CNodePtr &node);

Status ConstructCostGraphNodesByUniqueId(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root);

Status ConstructCostGraphNodesByUniqueIdTC(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root);

void ConstructCostGraphEdges(const std::vector<AnfNodePtr> &all_nodes);

void AugmentCostGraph(const std::vector<AnfNodePtr> &all_nodes);

void InferStraByTensorInfo(const TensorInfo &pre_out_tensor_info, Dimensions *stra);

Status ParallelStrategySearch(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root);

Status ParallelStrategyRecSearch(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root);

std::vector<std::vector<std::string>> RecInputTensorNames(const std::map<std::string, std::string>::iterator &it,
                                                          std::vector<std::vector<std::string>> input_tensor_names);
}  // namespace parallel
}  // namespace mindspore
#endif  // PARALLEL_STEP_AUTO_PARALLEL_H_
