/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_PASS_UTILS_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_PASS_UTILS_H_

#include <vector>
#include <string>
#include <utility>
#include <set>
#include <map>
#include <memory>
#include <unordered_map>
#include "base/base.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"

namespace mindspore {
namespace parallel {
constexpr char INTERLEAVED_OVERLAP_MATMUL[] = "interleaved_overlap_matmul";
constexpr char GRAD_OVERLAP_MATMUL[] = "grad_overlap_matmul";
constexpr char FORWARD_UNIQUE_ID_LIST[] = "forward_unique_id_list";
bool IsForwardNode(const CNodePtr &cnode);
bool IsDxMatMul(const CNodePtr &matmul_node);
bool IsDwMatMul(const CNodePtr &matmul_node);
void ExtractBackwardMatMul(const std::vector<CNodePtr> &origin_nodes_topological,
                           std::unordered_map<CNodePtr, CNodePtr> *backward_matmul_dx_dw_map);
std::string AnfNodeInfo(const AnfNodePtr &anf_node);
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_PASS_UTILS_H_
