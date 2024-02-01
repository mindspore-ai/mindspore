/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_PARALLEL_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_PARALLEL_H_

#include <vector>

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "utils/hash_map.h"
#include "frontend/optimizer/opt.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "pipeline/jit/ps/pipeline.h"
#include "frontend/parallel/ops_info/ops_utils.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"

using OperatorInfoPtr = std::shared_ptr<mindspore::parallel::OperatorInfo>;

namespace mindspore {
namespace parallel {
const uint64_t kUSecondInSecond = 1000000;
const int32_t RECURSION_LIMIT = 1000;

struct LossNodeInfo {
  bool has_tuple_getitem = false;
  int64_t dout_index = 0;  // now don't support the sens is a tuple
  CNodePtr loss_node = nullptr;
};

std::vector<AnfNodePtr> CreateInput(const Operator &op, const AnfNodePtr &node, const std::string &instance_name);
void ForwardCommunication(OperatorVector forward_op, const CNodePtr &node);

TensorLayout GetTensorInLayout(const AnfNodePtr &pre_node, int get_item_index);

void MarkForwardCNode(const FuncGraphPtr &root);

void SetVirtualDatasetStrategy(const CNodePtr &node);

// Create parallel operator for primitive node(has strategy)
void ExtractInformation(const std::vector<AnfNodePtr> &all_nodes);

// main step of Parallel
bool StepParallel(const FuncGraphPtr &root, const opt::OptimizerPtr &optimizer);

std::set<FuncGraphPtr> ForwardGraph(const FuncGraphPtr &root);

bool CreateGroupsByCkptFile(const std::string &file);

void InsertVirtualOutput(const FuncGraphPtr &root, const std::vector<AnfNodePtr> &all_nodes);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_STEP_PARALLEL_H_
