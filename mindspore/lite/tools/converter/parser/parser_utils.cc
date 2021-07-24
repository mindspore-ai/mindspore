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
#include "tools/converter/parser/parser_utils.h"
#include <memory>
#include <algorithm>
#include <vector>
#include <string>
#include "tools/converter/parser/tf_bidirection_gru_cf_fusion.h"
#include "tools/converter/parser/unused_node_remove_pass.h"
#include "tools/converter/parser/conv1d_inout_adjust.h"
#include "tools/converter/parser/inputs_adjust.h"
#include "ops/transpose.h"
#include "tools/converter/quant_param_holder.h"
#include "tools/common/tensor_util.h"
#include "tools/optimizer/common/gllo_utils.h"

namespace mindspore::lite {
namespace {
constexpr size_t kNumWeightIndex = 2;
}
void GetAllFuncGraph(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *all_func_graphs) {
  if (all_func_graphs->find(func_graph) == all_func_graphs->end()) {
    all_func_graphs->insert(func_graph);
  } else {
    return;
  }
  auto nodes = func_graph->nodes();
  for (auto &node : nodes) {
    if (IsValueNode<FuncGraph>(node)) {
      auto new_fg = (node->cast<ValueNodePtr>()->value())->cast<FuncGraphPtr>();
      GetAllFuncGraph(new_fg, all_func_graphs);
    }
    if (utils::isa<CNodePtr>(node)) {
      auto cnode = node->cast<CNodePtr>();
      for (auto &input : cnode->inputs()) {
        if (input->isa<ValueNode>()) {
          if (IsValueNode<FuncGraph>(input)) {
            auto new_fg = (input->cast<ValueNodePtr>()->value())->cast<FuncGraphPtr>();
            GetAllFuncGraph(new_fg, all_func_graphs);
          }
        }
      }
    }
  }
}

int CommonAnfAdjust(const std::set<FuncGraphPtr> &all_func_graphs) {
  for (auto func_graph : all_func_graphs) {
    {
      auto asylic_optimizer = std::make_shared<opt::GraphOptimizer>();
      auto asylic_pm = std::make_shared<opt::PassManager>("asylic pass manager", false);
      // fuse tf1.x bidirection_gru into GRU, must be placed here because graph is cyclic
      asylic_pm->AddPass(std::make_shared<opt::TfBidirectionGruCfFusion>());
      // remove remaining cyclic nodes
      asylic_pm->AddPass(std::make_shared<opt::UnusedNodeRemovePass>());
      asylic_optimizer->AddPassManager(asylic_pm);
      if (!asylic_optimizer->Optimize(func_graph)) {
        MS_LOG(ERROR) << "gru cf fusion pass failed.";
        ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
        return RET_ERROR;
      }
    }
    auto adjust_input = std::make_shared<InputAdjust>();
    if (!adjust_input->Run(func_graph)) {
      MS_LOG(ERROR) << "adjust input failed.";
      return RET_ERROR;
    }
    // adjust for conv1d
    auto conv1d_adjust = std::make_shared<Conv1DInOutAdjust>();
    if (!conv1d_adjust->Run(func_graph)) {
      MS_LOG(ERROR) << "adjust conv1d failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int GetTransposePerm(schema::Format src_format, schema::Format dst_format, std::vector<int> *perm) {
  MS_ASSERT(perm != nullptr);
  auto src_format_str = std::string(schema::EnumNameFormat(src_format));
  auto dst_format_str = std::string(schema::EnumNameFormat(dst_format));
  if (src_format_str.empty() || dst_format_str.empty() || src_format_str.size() != dst_format_str.size()) {
    MS_LOG(ERROR) << "src_format or dst_format is error.";
    return lite::RET_ERROR;
  }
  for (size_t i = 0; i < src_format_str.size(); ++i) {
    auto pos = src_format_str.find(dst_format_str[i]);
    if (pos == std::string::npos) {
      MS_LOG(ERROR) << "src_format and dst_format don't match.";
      return lite::RET_ERROR;
    }
    perm->push_back(static_cast<int>(pos));
  }
  return lite::RET_OK;
}
int GetTransposePermSharing(schema::Format src_format, schema::Format dst_format, std::vector<int> *perm) {
  MS_ASSERT(perm != nullptr);
  auto src_format_str = std::string(schema::EnumNameFormat(src_format));
  auto dst_format_str = std::string(schema::EnumNameFormat(dst_format));
  if (src_format_str.empty() || dst_format_str.empty() || src_format_str.size() != dst_format_str.size()) {
    MS_LOG(ERROR) << "src_format or dst_format is error.";
    return lite::RET_ERROR;
  }
  for (size_t i = 0; i < src_format_str.size(); ++i) {
    auto pos = dst_format_str.find(src_format_str[i]);
    if (pos == std::string::npos) {
      MS_LOG(ERROR) << "src_format and dst_format don't match.";
      return lite::RET_ERROR;
    }
    perm->push_back(static_cast<int>(pos));
  }
  return lite::RET_OK;
}

int TransposeInsertForWeightSharing(const FuncGraphPtr &graph, int64_t dst_format, int64_t format,
                                    const ParameterPtr &weight_node, std::vector<int> perm) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(weight_node != nullptr);
  auto node_list = TopoSort(graph->get_return());
  std::vector<CNodePtr> adjust_nodes;
  for (auto &node : node_list) {
    if (!utils::isa<CNodePtr>(node)) {
      continue;
    }
    if (opt::CheckPrimitiveType(node, prim::kPrimApplyMomentum) || opt::CheckPrimitiveType(node, prim::kPrimSGD) ||
        opt::CheckPrimitiveType(node, prim::kPrimAdam)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto inputs = cnode->inputs();
    if (std::any_of(inputs.begin(), inputs.end(),
                    [&](const AnfNodePtr &anf_node) { return weight_node == anf_node; })) {
      if (opt::CheckPrimitiveType(node, prim::kPrimConv2DFusion) ||
          opt::CheckPrimitiveType(node, opt::kPrimConv2DBackpropInputFusion) ||
          opt::CheckPrimitiveType(node, prim::kPrimConv2dTransposeFusion)) {
        auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
        prim->AddAttr(ops::kFormat, MakeValue<int64_t>(format));
        continue;
      }
      adjust_nodes.push_back(cnode);
    }
  }
  if (adjust_nodes.empty()) {
    MS_LOG(DEBUG) << "do not need to adjust nodes.";
    return lite::RET_OK;
  }
  auto perm_node = opt::BuildIntVecParameterNode(graph, perm, weight_node->fullname_with_scope() + "_sharing_perm");
  auto prim = std::make_shared<ops::Transpose>();
  prim->AddAttr("quant_params", std::make_shared<QuantParamHolder>(1, 1));
  prim->AddAttr(ops::kFormat, MakeValue<int64_t>(dst_format));
  auto transpose_node = graph->NewCNode(prim, {weight_node, perm_node});
  if (!weight_node->has_default()) {
    MS_LOG(DEBUG) << "Weight parameter should has default parameter.";
    return lite::RET_ERROR;
  }
  auto weight_tensor = weight_node->default_param()->cast<tensor::TensorPtr>();
  if (weight_tensor == nullptr) {
    MS_LOG(DEBUG) << "Default parameter of weight parameter should be a tensor.";
    return lite::RET_ERROR;
  }
  auto abstract = CreateTensorAbstract(weight_tensor->shape_c(), weight_tensor->data_type());
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstarct failed";
    return RET_ERROR;
  }
  transpose_node->set_abstract(abstract);
  transpose_node->set_fullname_with_scope(weight_node->fullname_with_scope() + "_sharing_post");
  for (auto &adjust_node : adjust_nodes) {
    auto inputs = adjust_node->inputs();
    std::replace_if(
      inputs.begin(), inputs.end(), [&weight_node](const AnfNodePtr &anf_node) { return weight_node == anf_node; },
      transpose_node);
    adjust_node->set_inputs(inputs);
  }
  return lite::RET_OK;
}

int HandleWeightSharing(const FuncGraphPtr &graph, int64_t format, const ParameterPtr &weight_node,
                        schema::Format src_format, schema::Format dst_format) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(weight_node != nullptr);
  if (src_format == dst_format) {
    return lite::RET_OK;
  }
  std::vector<int> perm;
  auto status = GetTransposePermSharing(src_format, dst_format, &perm);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "get perm failed.";
    return status;
  }
  status = TransposeInsertForWeightSharing(graph, dst_format, format, weight_node, perm);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "transpose insert failed.";
  }
  return status;
}

int TransposeInsertForWeightConst(const FuncGraphPtr &graph, const CNodePtr &conv_node, const CNodePtr &weight_node,
                                  std::vector<int> perm) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(weight_node != nullptr);
  auto manager = Manage(graph);
  if (opt::CheckPrimitiveType(weight_node, opt::kPrimIdentity) ||
      opt::CheckPrimitiveType(weight_node, prim::kPrimLoad)) {
    manager->Replace(weight_node, weight_node->input(1));
    return RET_OK;
  }
  auto perm_node = opt::BuildIntVecParameterNode(graph, perm, weight_node->fullname_with_scope() + "_const_perm");
  auto prim = std::make_shared<ops::Transpose>();
  prim->AddAttr("quant_params", std::make_shared<QuantParamHolder>(1, 1));
  auto transpose_node = graph->NewCNode(prim, {weight_node, perm_node});
  transpose_node->set_fullname_with_scope(weight_node->fullname_with_scope() + "_const_post");
  conv_node->set_input(kNumWeightIndex, transpose_node);
  return lite::RET_OK;
}

int HandleWeightConst(const FuncGraphPtr &graph, const CNodePtr &conv_node, const CNodePtr &weight_node,
                      schema::Format src_format, schema::Format dst_format) {
  MS_ASSERT(graph != nullptr);
  MS_ASSERT(weight_node != nullptr);
  if (src_format == dst_format) {
    return lite::RET_OK;
  }
  std::vector<int> perm;
  auto status = GetTransposePerm(src_format, dst_format, &perm);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "get perm failed.";
    return status;
  }
  status = TransposeInsertForWeightConst(graph, conv_node, weight_node, perm);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "transpose insert failed.";
  }
  return status;
}
}  // namespace mindspore::lite
