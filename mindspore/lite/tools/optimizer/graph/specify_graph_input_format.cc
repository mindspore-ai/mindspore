/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/graph/specify_graph_input_format.h"
#include <memory>
#include <vector>
#include <stack>
#include <set>
#include "mindspore/core/ops/array_ops.h"
#include "mindspore/core/ops/framework_ops.h"
#include "tools/converter/parser/parser_utils.h"
#include "tools/optimizer/common/format_utils.h"
#include "src/common/log_adapter.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "ops/transpose.h"

namespace mindspore {
namespace opt {
bool SpecifyGraphInputFormat::Run(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  if (exp_graph_input_format_ == cur_graph_input_format_) {
    return true;
  }
  if ((exp_graph_input_format_ != mindspore::NHWC && exp_graph_input_format_ != mindspore::NCHW) ||
      (cur_graph_input_format_ != mindspore::NHWC && cur_graph_input_format_ != mindspore::NCHW)) {
    MS_LOG(ERROR) << "this pass only support to transfer graph input format between nhwc with nchw.";
    return false;
  }
  auto manager = Manage(graph);
  MS_CHECK_TRUE_MSG(manager != nullptr, false, "manager is nullptr.");
  if (HandleGraphInput(graph) != lite::RET_OK) {
    MS_LOG(ERROR) << "Specify graph-input format failed.";
    return false;
  }
  return true;
}

STATUS SpecifyGraphInputFormat::HandleGraphInput(const FuncGraphPtr &graph) {
  MS_ASSERT(graph != nullptr);
  auto manager = graph->manager();
  MS_ASSERT(manager != nullptr);
  auto graph_inputs = graph->get_inputs();
  for (const auto &input : graph_inputs) {
    auto input_node = input->cast<ParameterPtr>();
    MS_ASSERT(input_node != nullptr);
    auto abstract = input_node->abstract();
    MS_CHECK_TRUE_MSG(abstract != nullptr, lite::RET_NULL_PTR, "abstract is nullptr");

    ShapeVector shape;
    if (FetchShapeFromAbstract(abstract, &shape) != lite::RET_OK) {
      MS_LOG(ERROR) << "fetch shape failed." << input->fullname_with_scope();
      return lite::RET_ERROR;
    }
    if (shape.size() != kInputSizeFour) {
      continue;
    }
    ShapeVector transfer_shape;
    if (exp_graph_input_format_ == mindspore::NCHW) {
      transfer_shape = {shape[0], shape[kInputIndexThree], shape[1], shape[kInputIndexTwo]};
    } else {
      transfer_shape = {shape[0], shape[kInputIndexTwo], shape[kInputIndexThree], shape[1]};
    }
    CNodePtr trans_cnode;
    if (exp_graph_input_format_ == mindspore::NCHW) {
      trans_cnode = opt::GenTransposeNode(graph, input, kNC2NH, input->fullname_with_scope() + "_nc2nh");
    } else {
      trans_cnode = opt::GenTransposeNode(graph, input, kNH2NC, input->fullname_with_scope() + "_nh2nc");
    }
    if (trans_cnode == nullptr) {
      MS_LOG(ERROR) << "create transpose cnode failed.";
      return lite::RET_ERROR;
    }
    auto trans_prim = GetValueNode<PrimitivePtr>(trans_cnode->input(0));
    MS_CHECK_TRUE_MSG(trans_prim != nullptr, lite::RET_NULL_PTR, "GetValueNode Failed");
    if (exp_graph_input_format_ == mindspore::NCHW) {
      trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NCHW));
    } else {
      trans_prim->AddAttr(ops::kFormat, MakeValue<int64_t>(NHWC));
    }
    trans_cnode->set_abstract(abstract->Clone());
    abstract->set_shape(std::make_shared<abstract::Shape>(transfer_shape));
    (void)manager->Replace(input, trans_cnode);
  }
  return lite::RET_OK;
}

bool CheckInputsFormatNHWC(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    manager = Manage(func_graph, true);
    MS_CHECK_TRUE_RET(manager != nullptr, {});
    std::set<FuncGraphPtr> all_func_graphs;
    lite::GetAllFuncGraph(func_graph, &all_func_graphs);
    for (auto &graph : all_func_graphs) {
      manager->AddFuncGraph(graph);
    }
  }

  auto node_users = manager->node_users();
  std::vector<AnfNodePtr> nodes;
  auto inputs = func_graph->get_inputs();
  (void)std::for_each(inputs.begin(), inputs.end(), [&nodes](const AnfNodePtr &input) {
    if (opt::GetAnfNodeOutputShape(input, 0).size() == DIMENSION_4D) {
      nodes.push_back(input);
    }
  });
  for (auto input : nodes) {
    auto itr = node_users.find(input);
    for (auto pair : itr->second) {
      auto used_node = pair.first;
      MS_CHECK_TRUE_RET(used_node != nullptr && used_node->isa<CNode>(), false);
      if (!opt::CheckPrimitiveType(used_node, prim::kPrimTranspose)) {
        return false;
      }
      std::vector<int> perm;
      if (GetTransposePerm(used_node->cast<CNodePtr>(), &perm) != RET_OK) {
        MS_LOG(ERROR) << "fetch transpose perm failed.";
        return false;
      }
      if (perm != kNH2NC) {
        return false;
      }
    }
  }
  return true;
}

std::vector<AnfNodePtr> GetTracedCnodes(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_RET(manager != nullptr, {});
  auto node_users = manager->node_users();
  auto nhwc_ops = GetNHWCOpMap();
  std::stack<AnfNodePtr> nodes;
  for (auto input : func_graph->get_inputs()) {
    if (opt::GetAnfNodeOutputShape(input, 0).size() == DIMENSION_4D) {
      nodes.push(input);
    }
  }

  std::vector<AnfNodePtr> traced_nodes;
  std::vector<AnfNodePtr> checked_nodes;
  while (!nodes.empty()) {
    auto node = nodes.top();
    nodes.pop();
    if (std::find(checked_nodes.begin(), checked_nodes.end(), node) != checked_nodes.end() ||
        opt::CheckPrimitiveType(node, prim::kPrimReturn)) {
      continue;
    }
    if (node->isa<CNode>()) {
      auto cnode = node->cast<CNodePtr>();
      MS_CHECK_TRUE_RET(cnode != nullptr, {});
      MS_CHECK_TRUE_RET(cnode->size() > 0, {});
      if (cnode->size() > 1) {
        auto input_node = cnode->input(1);
        auto itr = std::find(traced_nodes.begin(), traced_nodes.end(), input_node);
        if (itr != traced_nodes.end()) {
          (void)traced_nodes.erase(itr + 1, traced_nodes.end());
        }
      }
      auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
      if (prim != nullptr && nhwc_ops.find(prim->name()) != nhwc_ops.end()) {
        return traced_nodes;
      }
      traced_nodes.push_back(node);
    }
    auto itr = node_users.find(node);
    MS_CHECK_TRUE_RET(itr != node_users.end(), {});
    for (auto &pair : itr->second) {
      nodes.push(pair.first);
    }
    checked_nodes.push_back(node);
  }
  return {};
}

bool SpecifyGraphInputFormat::GetCurGraphInputFormat(const FuncGraphPtr &func_graph, converter::FmkType fmk_type,
                                                     mindspore::Format *input_format) {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(input_format != nullptr);
  if (fmk_type == converter::kFmkTypeTf || fmk_type == converter::kFmkTypeTflite) {
    *input_format = NHWC;
  } else {
    *input_format = NCHW;
  }

  if (CheckInputsFormatNHWC(func_graph)) {
    *input_format = NHWC;
    return true;
  }
  auto traced_nodes = GetTracedCnodes(func_graph);
  for (auto node : traced_nodes) {
    if (opt::CheckPrimitiveType(node, prim::kPrimTranspose)) {
      auto cnode = node->cast<CNodePtr>();
      MS_CHECK_TRUE_RET(cnode != nullptr, false);
      std::vector<int> perm;
      if (GetTransposePerm(cnode, &perm) != RET_OK) {
        MS_LOG(ERROR) << "fetch transpose perm failed.";
        return false;
      }
      if (perm == kNC2NH) {
        *input_format = NCHW;
        return true;
      } else if (perm == kNH2NC) {
        *input_format = NHWC;
        return true;
      }
    }
  }
  return true;
}
}  // namespace opt
}  // namespace mindspore
