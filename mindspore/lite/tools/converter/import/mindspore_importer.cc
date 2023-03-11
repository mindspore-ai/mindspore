/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/converter/import/mindspore_importer.h"
#include <memory>
#include <map>
#include <set>
#include <vector>
#include <regex>
#include <queue>
#include <algorithm>
#include "tools/converter/parser/parser_utils.h"
#include "tools/converter/import/cast_op_adjust.h"
#include "tools/converter/import/primitive_adjust.h"
#include "tools/converter/import/mindir_adjust.h"
#include "tools/converter/import/mindir_control_flow_adjust.h"
#include "tools/converter/import/remove_public_primitive.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/common/string_util.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/parser/unify_format.h"
#include "tools/converter/parser/lstm_adjust_pass.h"
#include "tools/optimizer/graph/redundant_op_remove_pass.h"
#include "nnacl/op_base.h"
#include "src/common/common.h"

namespace mindspore::lite {
namespace {
constexpr size_t kConvWeightIndex = 2;
constexpr size_t kDependInputNum = 3;
constexpr size_t kDependFirstInputIdx = 1;
constexpr size_t kTupleGetItemFirstInputIdx = 1;
}  // namespace
STATUS MindsporeImporter::Mindir2AnfAdjust(const FuncGraphPtr &func_graph,
                                           const std::shared_ptr<ConverterPara> &param) {
  MS_ASSERT(func_graph != nullptr);
  auto primitive_adjust_pass = std::make_shared<PrimitiveAdjust>();
  MS_CHECK_TRUE_MSG(primitive_adjust_pass != nullptr, RET_NULL_PTR, "primitive_adjust_pass is nullptr.");
  primitive_adjust_pass->SetFmkType(param->fmk_type);
  if (!primitive_adjust_pass->Run(func_graph)) {
    MS_LOG(ERROR) << "primitive adjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  bool is_optimized = false;
  auto value = func_graph->get_attr(kIsOptimized);
  if (value != nullptr) {
    is_optimized = GetValue<bool>(value);
  }
  if (!is_optimized) {
    auto mindir_adjust_pass = std::make_shared<MindirAdjust>();
    MS_CHECK_TRUE_MSG(mindir_adjust_pass != nullptr, RET_NULL_PTR, "mindir_adjust_pass is nullptr.");
    mindir_adjust_pass->SetFmkType(param->fmk_type);
    mindir_adjust_pass->SetTrainFlag(param->train_model);
    if (!mindir_adjust_pass->Run(func_graph)) {
      MS_LOG(ERROR) << "MindIr adjust failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return RET_ERROR;
    }
  }
  if (!param->train_model) {
    auto cast_op_adjust = std::make_shared<opt::CastOpAdjust>();
    MS_CHECK_TRUE_MSG(cast_op_adjust != nullptr, RET_NULL_PTR, "cast_op_adjust is nullptr.");
    if (!cast_op_adjust->Run(func_graph, param->device.find("Ascend") != std::string::npos)) {
      MS_LOG(ERROR) << "MindIr adjust cast operator failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return RET_ERROR;
    }
  }
  auto mindir_control_flow_adjust = std::make_shared<MindIRControlFlowAdjust>();
  MS_CHECK_TRUE_MSG(mindir_control_flow_adjust != nullptr, RET_NULL_PTR, "mindir_control_flow_adjust is nullptr.");
  mindir_control_flow_adjust->SetFmkType(param->fmk_type);
  if (!mindir_control_flow_adjust->Run(func_graph)) {
    MS_LOG(ERROR) << "MindIR control flow adjust failed.";
    ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
    return RET_ERROR;
  }
  return RET_OK;
}

namespace {
bool IsEmptyOp(const AnfNodePtr &node) {
  MS_ASSERT(node != nullptr);
  return (opt::CheckPrimitiveType(node, prim::kPrimMakeTuple) || opt::CheckPrimitiveType(node, prim::kPrimReturn) ||
          opt::CheckPrimitiveType(node, prim::kPrimTupleGetItem) || opt::CheckPrimitiveType(node, prim::kPrimDepend) ||
          opt::CheckPrimitiveType(node, prim::kPrimUpdateState) || opt::CheckPrimitiveType(node, prim::kPrimLoad));
}

void RemovePostEdgeOfParameter(const AnfNodePtr &parameter) {
  MS_ASSERT(parameter != nullptr);
  auto func_graph = parameter->func_graph();
  MS_ASSERT(func_graph != nullptr);
  auto manager = Manage(func_graph);
  MS_ASSERT(maneger != nullptr);
  auto nodes_users = manager->node_users();
  auto node_users_iter = nodes_users.find(parameter);
  MS_ASSERT(node_users_iter != nodes_users.end());
  for (const auto &node_user_iter : node_users_iter->second) {
    MS_ASSERT(utils::isa<CNodePtr>(node_user_iter.first));
    auto node_user_cnode = utils::cast<CNodePtr>(node_user_iter.first);
    auto &node_user_cnode_inputs = node_user_cnode->inputs();
    std::vector<AnfNodePtr> new_node_user_cnode_inputs;
    for (size_t i = 0; i < node_user_cnode_inputs.size(); i++) {
      if (static_cast<int>(i) == node_user_iter.second) {
        continue;
      }
      new_node_user_cnode_inputs.emplace_back(node_user_cnode_inputs.at(i));
    }
    node_user_cnode->set_inputs(new_node_user_cnode_inputs);
  }
}
}  // namespace

void MindsporeImporter::RemoveUnusedGraphInput(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  // drop unused input_parameter and disconnect edge
  auto graph_inputs = func_graph->get_inputs();
  std::vector<AnfNodePtr> graph_outputs;
#if !defined(_WIN32) && !defined(_WIN64)
  auto graph_outputs_index = opt::GetNodeInputs(func_graph->get_return());
  std::transform(graph_outputs_index.begin(), graph_outputs_index.end(), std::back_inserter(graph_outputs),
                 [](const auto &item) { return item.first; });
#endif
  auto manager = Manage(func_graph);
  MS_ASSERT(manager != nullptr);
  auto nodes_users = manager->node_users();
  std::vector<AnfNodePtr> unused_inputs;
  for (const auto &input : graph_inputs) {
    bool found_used = false;
    std::queue<AnfNodePtr> q;
    q.push(input);
    while (!q.empty()) {
      auto cur_node = q.front();
      q.pop();
      if (cur_node != input && !IsEmptyOp(cur_node)) {
        found_used = true;
        break;
      }
      auto node_users_itr = nodes_users.find(cur_node);
      if (node_users_itr == nodes_users.end()) {
        continue;
      }
      for (const auto &node_user_itr : node_users_itr->second) {
        MS_ASSERT(utils::isa<CNodePtr>(node_user_itr.first));
        auto node_user_cnode = utils::cast<CNodePtr>(node_user_itr.first);
        q.push(node_user_cnode);
      }
    }
    if (std::find(graph_outputs.begin(), graph_outputs.end(), input) != graph_outputs.end()) {
      found_used = true;
    }
    if (!found_used) {
      if (nodes_users.find(input) != nodes_users.end()) {
        RemovePostEdgeOfParameter(input);
      }
      unused_inputs.push_back(input);
    }
  }
  for (auto &input : unused_inputs) {
    func_graph->DropNode(input);
  }
}
}  // namespace mindspore::lite
