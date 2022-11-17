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
#include "tools/converter/parser/parser_utils.h"
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <unordered_map>
#include "ops/adam.h"
#include "ops/apply_momentum.h"
#include "ops/fusion/conv2d_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "ops/fusion/conv2d_backprop_input_fusion.h"
#include "ops/sgd.h"
#include "tools/common/tensor_util.h"
#include "tools/converter/parser/conv1d_inout_adjust.h"
#include "tools/converter/parser/inputs_adjust.h"
#include "tools/converter/parser/tf_bidirection_gru_cf_fusion.h"
#include "tools/converter/parser/unused_node_remove_pass.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/optimizer/format/to_format_base.h"
#include "tools/optimizer/common/pass_manager_extends.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"
#include "src/common/common.h"
#include "tools/converter/parser/conv2d_transpose_input_adjust.h"

namespace mindspore::lite {
namespace {
std::unordered_map<std::string, size_t> weight_indexs = {{ops::kNameConv2DFusion, 2},
                                                         {ops::kNameConv2DBackpropInputFusion, 2},
                                                         {ops::kNameConv2dTransposeFusion, 2},
                                                         {ops::kNameApplyMomentum, 1},
                                                         {ops::kNameSGD, 1},
                                                         {ops::kNameAdam, 1}};
}  // namespace

void GetAllFuncGraph(const FuncGraphPtr &func_graph, std::set<FuncGraphPtr> *all_func_graphs) {
  MS_ASSERT(all_func_graphs != nullptr);
  MS_ASSERT(func_graph != nullptr);
  if (all_func_graphs->find(func_graph) == all_func_graphs->end()) {
    all_func_graphs->insert(func_graph);
  } else {
    return;
  }
  auto nodes = TopoSort(func_graph->get_return());
  for (auto &node : nodes) {
    if (IsValueNode<FuncGraph>(node)) {
      MS_ASSERT(node->cast<ValueNodePtr>() != nullptr);
      MS_ASSERT(node->cast<ValueNodePtr>()->value() != nullptr);
      MS_ASSERT((node->cast<ValueNodePtr>()->value())->cast<FuncGraphPtr>() != nullptr);
      auto new_fg = (node->cast<ValueNodePtr>()->value())->cast<FuncGraphPtr>();
      GetAllFuncGraph(new_fg, all_func_graphs);
    }
    if (utils::isa<CNodePtr>(node)) {
      auto cnode = node->cast<CNodePtr>();
      MS_ASSERT(cnode != nullptr);
      for (auto &input : cnode->inputs()) {
        if (input->isa<ValueNode>()) {
          if (IsValueNode<FuncGraph>(input)) {
            MS_ASSERT(input->cast<ValueNodePtr>() != nullptr);
            MS_ASSERT(input->cast<ValueNodePtr>()->value() != nullptr);
            MS_ASSERT((input->cast<ValueNodePtr>()->value())->cast<FuncGraphPtr>() != nullptr);
            auto new_fg = (input->cast<ValueNodePtr>()->value())->cast<FuncGraphPtr>();
            GetAllFuncGraph(new_fg, all_func_graphs);
          }
        }
      }
    }
  }
}

int CommonAnfAdjust(const FuncGraphPtr &func_graph) {
  MS_ASSERT(func_graph != nullptr);
  bool is_optimized = false;
  auto value = func_graph->get_attr(kIsOptimized);
  if (value != nullptr) {
    is_optimized = GetValue<bool>(value);
  }
  {
    auto asylic_optimizer = std::make_shared<opt::GraphOptimizer>();
    MS_CHECK_TRUE_MSG(asylic_optimizer != nullptr, RET_NULL_PTR, "asylic_optimizer is nullptr.");
    auto asylic_pm = std::make_shared<opt::LitePassManager>("asylic pass manager", false);
    MS_CHECK_TRUE_MSG(asylic_pm != nullptr, RET_NULL_PTR, "asylic_pm is nullptr.");

    // fuse tf1.x bidirection_gru into GRU, must be placed here because graph is cyclic
    asylic_pm->AddPass(std::make_shared<opt::TfBidirectionGruCfFusion>());
    // remove remaining cyclic nodes
    asylic_pm->AddPass(std::make_shared<opt::UnusedNodeRemovePass>());
    asylic_optimizer->AddPassManager(asylic_pm);

    if (!is_optimized && !asylic_optimizer->Optimize(func_graph)) {
      MS_LOG(ERROR) << "gru cf fusion pass failed.";
      ReturnCode::GetSingleReturnCode()->UpdateReturnCode(RET_ERROR);
      return RET_ERROR;
    }
  }

  std::set<FuncGraphPtr> all_func_graphs = {};
  GetAllFuncGraph(func_graph, &all_func_graphs);
  for (auto sub_graph : all_func_graphs) {
    auto adjust_input = std::make_shared<InputAdjust>();
    MS_CHECK_TRUE_MSG(adjust_input != nullptr, RET_NULL_PTR, "adjust_input is nullptr.");
    if (!adjust_input->Run(sub_graph)) {
      MS_LOG(ERROR) << "adjust input failed.";
      return RET_ERROR;
    }
    // adjust for conv1d
    if (!is_optimized) {
      auto conv1d_adjust = std::make_shared<Conv1DInOutAdjust>();
      MS_CHECK_TRUE_MSG(conv1d_adjust != nullptr, RET_NULL_PTR, "conv1d_adjust is nullptr.");
      if (!conv1d_adjust->Run(sub_graph)) {
        MS_LOG(ERROR) << "adjust conv1d failed.";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

int GetTransposePerm(schema::Format src_format, schema::Format dst_format, std::vector<int> *perm) {
  MS_CHECK_TRUE_MSG(perm != nullptr, RET_NULL_PTR, "perm is nullptr.");
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
  MS_CHECK_TRUE_MSG(perm != nullptr, RET_NULL_PTR, "perm is nullptr.");
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

AnfNodePtr GetRealConvWeightNode(const FuncGraphPtr &graph, const CNodePtr &cnode, size_t index) {
  MS_CHECK_TRUE_MSG(graph != nullptr, nullptr, "graph is nullptr.");
  MS_CHECK_TRUE_MSG(cnode != nullptr, nullptr, "cnode is nullptr.");
  auto weight_node = cnode->input(index);
  MS_CHECK_TRUE_MSG(weight_node != nullptr, nullptr, "weight_node is nullptr.");
  bool is_real_weight =
    !opt::CheckPrimitiveType(weight_node, opt::kPrimIdentity) && !opt::CheckPrimitiveType(weight_node, prim::kPrimLoad);
  while (!is_real_weight) {
    if (!utils::isa<CNode>(weight_node)) {
      MS_LOG(ERROR) << "weight node is invalid.";
      return nullptr;
    }
    auto weight_cnode = weight_node->cast<CNodePtr>();
    weight_node = weight_cnode->input(1);
    MS_CHECK_TRUE_MSG(weight_node != nullptr, nullptr, "weight_node is nullptr.");
    is_real_weight = !opt::CheckPrimitiveType(weight_node, opt::kPrimIdentity) &&
                     !opt::CheckPrimitiveType(weight_node, prim::kPrimLoad);
  }
  auto manager = Manage(graph);
  MS_CHECK_TRUE_MSG(manager != nullptr, nullptr, "manager is nullptr.");
  if (!manager->Replace(cnode->input(index), weight_node)) {
    MS_LOG(ERROR) << "Replace weight node failed.";
    return nullptr;
  }
  return weight_node;
}

int UnifyConvWeightFormat(const FuncGraphPtr &graph, const CNodePtr &cnode, schema::Format src_format,
                          schema::Format dst_format, std::set<AnfNodePtr> *has_visited) {
  MS_CHECK_TRUE_MSG(graph != nullptr, RET_NULL_PTR, "graph is nullptr.");
  MS_CHECK_TRUE_MSG(cnode != nullptr, RET_NULL_PTR, "cnode is nullptr.");
  MS_CHECK_TRUE_MSG(has_visited != nullptr, RET_NULL_PTR, "has_visited is nullptr.");
  if (src_format == dst_format) {
    return lite::RET_OK;
  }
  auto primitive_ptr = GetValueNode<PrimitivePtr>(cnode->input(0));
  auto primitive_name = primitive_ptr->name();
  if (weight_indexs.find(primitive_name) == weight_indexs.end()) {
    MS_LOG(ERROR) << primitive_name << " is not a member of convolution's family.";
    return RET_ERROR;
  }
  size_t index = weight_indexs[primitive_name];
  if (GetRealConvWeightNode(graph, cnode, index) == nullptr) {
    MS_LOG(ERROR) << "current conv node is invalid, node name is " << cnode->fullname_with_scope();
    return RET_ERROR;
  }
  bool is_const_weight = true;
  auto weight_node = cnode->input(index);
  MS_CHECK_TRUE_MSG(weight_node != nullptr, RET_NULL_PTR, "weight_node is nullptr.");
  if (utils::isa<CNode>(weight_node)) {
    is_const_weight = false;
  } else if (utils::isa<Parameter>(weight_node)) {
    auto weight_param_node = weight_node->cast<ParameterPtr>();
    MS_CHECK_TRUE_MSG(weight_param_node != nullptr, RET_NULL_PTR, "weight_param_node is nullptr.");
    if (!weight_param_node->has_default()) {
      is_const_weight = false;
    }
  }
  int status;
  if (is_const_weight) {
    status = UnifyConstConvWeight(graph, weight_node, src_format, dst_format, has_visited);
  } else {
    status = UnifyVariableConvWeight(graph, weight_node, src_format, dst_format, has_visited);
  }
  if (status != RET_OK) {
    MS_LOG(ERROR) << "unfiy coneight failed, cnode name is " << cnode->fullname_with_scope();
  }
  return status;
}

int UnifyVariableConvWeight(const FuncGraphPtr &graph, const AnfNodePtr &weight_node, schema::Format src_format,
                            schema::Format dst_format, std::set<AnfNodePtr> *has_visited) {
  MS_CHECK_TRUE_MSG(graph != nullptr, RET_NULL_PTR, "graph is nullptr.");
  MS_CHECK_TRUE_MSG(weight_node != nullptr, RET_NULL_PTR, "weight_node is nullptr.");
  MS_CHECK_TRUE_MSG(has_visited != nullptr, RET_NULL_PTR, "has_visited is nullptr.");
  if (src_format == dst_format) {
    return lite::RET_OK;
  }
  std::vector<int> perm;
  auto status = GetTransposePerm(src_format, dst_format, &perm);
  if (status != lite::RET_OK) {
    MS_LOG(ERROR) << "get perm failed.";
    return status;
  }
  auto manager = Manage(graph);
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_NULL_PTR, "manager is nullptr.");
  CNodePtr trans_cnode = nullptr;
  auto weight_node_users = manager->node_users()[weight_node];
  for (auto &weight_node_user : weight_node_users) {
    auto post_node = weight_node_user.first;
    if (!utils::isa<CNodePtr>(post_node)) {
      MS_LOG(ERROR) << "post node is invalid.";
      return RET_ERROR;
    }
    if (!opt::ToFormatBase::IsWeightNodeSensitive(post_node)) {
      continue;
    }
    has_visited->insert(post_node);
    if (trans_cnode == nullptr) {
      trans_cnode = opt::GenTransposeNode(graph, weight_node, perm, weight_node->fullname_with_scope() + "_post_perm");
      MS_CHECK_TRUE_MSG(trans_cnode != nullptr, RET_NULL_PTR, "trans_cnode is nullptr.");
      auto abstract = weight_node->abstract();
      ShapeVector shape;
      if (abstract != nullptr) {
        ShapeVector weight_shape;
        if (opt::FetchShapeFromAbstract(abstract, &weight_shape) != RET_OK) {
          MS_LOG(ERROR) << "fetch shape from abstract failed.";
          return RET_ERROR;
        }
        if (!weight_shape.empty()) {
          if (weight_shape.size() != opt::kInputSizeFour) {
            MS_LOG(ERROR) << "conv weight shape is invalid, which is not 4D, now is " << weight_shape.size();
            return RET_ERROR;
          }
          std::transform(perm.begin(), perm.end(), std::back_inserter(shape),
                         [&weight_shape](const int index) { return weight_shape[index]; });
        }
        abstract = abstract->Clone();
      } else {
        abstract = CreateTensorAbstract(shape, TypeId::kNumberTypeFloat32);
        MS_CHECK_TRUE_MSG(abstract != nullptr, RET_NULL_PTR, "abstract is nullptr.");
      }
      auto shape_ptr = std::make_shared<abstract::Shape>(shape);
      MS_CHECK_TRUE_MSG(shape_ptr != nullptr, RET_NULL_PTR, "shape_ptr is nullptr.");
      abstract->set_shape(shape_ptr);
      trans_cnode->set_abstract(abstract);
    }
    auto post_cnode = post_node->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(post_cnode != nullptr, RET_NULL_PTR, "post_cnode is nullptr.");
    auto tr = manager->Transact();
    tr.SetEdge(post_cnode, weight_node_user.second, trans_cnode);
    tr.Commit();
  }
  return RET_OK;
}

int UnifyConstConvWeight(const FuncGraphPtr &graph, const AnfNodePtr &weight_node, schema::Format src_format,
                         schema::Format dst_format, std::set<AnfNodePtr> *has_visited) {
  MS_CHECK_TRUE_MSG(graph != nullptr, RET_NULL_PTR, "graph is nullptr.");
  MS_CHECK_TRUE_MSG(weight_node != nullptr, RET_NULL_PTR, "weight_node is nullptr.");
  MS_CHECK_TRUE_MSG(has_visited != nullptr, RET_NULL_PTR, "has_visited is nullptr.");
  if (src_format == dst_format) {
    return lite::RET_OK;
  }
  auto weight_value = opt::GetTensorInfo(weight_node);
  if (weight_value == nullptr) {
    MS_LOG(ERROR) << "conv weight is non-const.";
    return RET_ERROR;
  }
  if (weight_value->shape().size() != kShape4dDims) {
    return lite::RET_OK;
  }
  auto status = opt::TransFilterFormat(weight_value, src_format, dst_format);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "TransFilter " << EnumNameFormat(src_format) << "To" << EnumNameFormat(dst_format)
                  << " failed, node : " << weight_node->fullname_with_scope();
    return RET_ERROR;
  }
  auto type_id = static_cast<TypeId>(weight_value->data_type());
  auto shape = weight_value->shape();
  auto abstract = CreateTensorAbstract(shape, type_id);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstarct failed";
    return RET_ERROR;
  }
  weight_node->set_abstract(abstract);
  if (HandleConstConvWeightShared(graph, weight_node, src_format, dst_format, has_visited) != RET_OK) {
    MS_LOG(ERROR) << "handle const conv weight-shared failed, node name is " << weight_node->fullname_with_scope();
    return RET_ERROR;
  }
  return RET_OK;
}

int HandleConstConvWeightShared(const FuncGraphPtr &graph, const AnfNodePtr &weight_node, schema::Format src_format,
                                schema::Format dst_format, std::set<AnfNodePtr> *has_visited) {
  MS_CHECK_TRUE_MSG(graph != nullptr, RET_NULL_PTR, "graph is nullptr.");
  MS_CHECK_TRUE_MSG(weight_node != nullptr, RET_NULL_PTR, "weight_node is nullptr.");
  MS_CHECK_TRUE_MSG(has_visited != nullptr, RET_NULL_PTR, "has_visited is nullptr.");
  if (src_format == dst_format) {
    return RET_OK;
  }
  std::vector<int> perm;
  auto status = GetTransposePermSharing(src_format, dst_format, &perm);
  if (status != RET_OK) {
    MS_LOG(ERROR) << "get perm failed.";
    return status;
  }
  auto manager = Manage(graph);
  MS_CHECK_TRUE_MSG(manager != nullptr, RET_NULL_PTR, "manager is nullptr.");
  CNodePtr trans_cnode = nullptr;
  auto weight_node_users = manager->node_users()[weight_node];
  for (auto &weight_node_user : weight_node_users) {
    auto post_node = weight_node_user.first;
    if (!utils::isa<CNodePtr>(post_node)) {
      MS_LOG(ERROR) << "post node is invalid.";
      return RET_ERROR;
    }
    if (opt::ToFormatBase::IsWeightNodeSensitive(post_node)) {
      has_visited->insert(post_node);
      continue;
    }
    if (trans_cnode == nullptr) {
      trans_cnode = opt::GenTransposeNode(graph, weight_node, perm, weight_node->fullname_with_scope() + "_post_perm");
      MS_CHECK_TRUE_MSG(trans_cnode != nullptr, RET_NULL_PTR, "trans_cnode is nullptr.");
      auto prim = GetValueNode<PrimitivePtr>(trans_cnode->input(0));
      MS_CHECK_TRUE_MSG(prim != nullptr, RET_NULL_PTR, "prim is nullptr.");
      (void)prim->AddAttr(ops::kFormat, MakeValue<int64_t>(dst_format));
      auto weight_value = opt::GetTensorInfo(weight_node);
      MS_CHECK_TRUE_MSG(weight_value != nullptr, RET_NULL_PTR, "weight_value is nullptr.");

      auto weight_shape = weight_value->shape();
      ShapeVector shape;
      if (!weight_shape.empty()) {
        if (weight_shape.size() != opt::kInputSizeFour) {
          MS_LOG(ERROR) << "conv weight shape is invalid, which is not 4D, now is " << weight_shape.size();
          return RET_ERROR;
        }
        std::transform(perm.begin(), perm.end(), std::back_inserter(shape),
                       [&weight_shape](const int index) { return weight_shape[index]; });
      }
      auto abstract = weight_node->abstract();
      MS_CHECK_TRUE_MSG(abstract != nullptr, RET_NULL_PTR, "abstract is nullptr.");
      abstract = abstract->Clone();
      auto shape_ptr = std::make_shared<abstract::Shape>(shape);
      MS_CHECK_TRUE_MSG(shape_ptr != nullptr, RET_NULL_PTR, "shape_ptr is nullptr.");
      abstract->set_shape(shape_ptr);
      trans_cnode->set_abstract(abstract);
    }
    auto post_cnode = post_node->cast<CNodePtr>();
    MS_CHECK_TRUE_MSG(post_cnode != nullptr, RET_NULL_PTR, "post_cnode is nullptr.");
    auto tr = manager->Transact();
    tr.SetEdge(post_cnode, weight_node_user.second, trans_cnode);
    tr.Commit();
  }
  return RET_OK;
}
}  // namespace mindspore::lite
