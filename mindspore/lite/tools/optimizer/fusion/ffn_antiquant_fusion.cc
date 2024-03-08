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
#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/ffn_antiquant_fusion.h"
#include <vector>
#include <memory>
#include "mindspore/core/ops/array_ops.h"
#include "ir/func_graph.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/op_name.h"
#include "ops/other_ops.h"
#include "mindspore/lite/tools/optimizer/common/gllo_utils.h"
#include "mindspore/lite/tools/converter/quantizer/quantize_util.h"

namespace mindspore {
namespace {
constexpr size_t kFFNAntiquantScaleInputIndex = 10;
constexpr size_t kFFNWeight1InputIndex = 1;
constexpr size_t kFFNWeight2InputIndex = 2;
constexpr char IN_STRATEGY[] = "in_strategy";
}  // namespace
namespace opt {
bool FFNAntiquantFusion::Run(const FuncGraphPtr &func_graph) {
  CHECK_NULL_RETURN(func_graph);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node) || !CheckPrimitiveType(node, prim::kPrimFFN)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (Process(func_graph, cnode) != lite::RET_OK) {
      MS_LOG(ERROR) << "Do FFNAntiquantFusion failed, node name is " << node->fullname_with_scope();
      return false;
    }
  }
  return true;
}

// Origin Param Shape: MOE: [1, 1, N], FFN: [1, N]
// Repeat Param Shape: MOE: [R, N], FFN: [1, N]
ParameterPtr FFNAntiquantFusion::RepeatParameter(const FuncGraphPtr &func_graph, const ParameterPtr param_node,
                                                 int repeat_times) {
  if (repeat_times == 1) {
    return param_node;
  }
  if (!(param_node)->has_default()) {
    MS_LOG(ERROR) << "param_node: " << param_node->fullname_with_scope() << " don't have default param";
    return nullptr;
  }
  ShapeVector ori_shape;
  if (opt::FetchShapeFromAbstract(param_node->abstract(), &ori_shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "fetch shape failed." << param_node->fullname_with_scope();
    return nullptr;
  }
  if (ori_shape[0] != 1) {
    MS_LOG(ERROR) << "Only support param shape[0] is 1, but the param shape is " << ori_shape
                  << ", only support [1, 1, N] or [1, N]";
    return nullptr;
  }

  auto tensor = std::static_pointer_cast<tensor::Tensor>(param_node->default_param());
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "default_param can not cast to tensor::Tensor";
    return nullptr;
  }

  TypeId type_id;
  if (opt::GetDataTypeFromAnfNode(param_node, &type_id) != RET_OK) {
    MS_LOG(WARNING) << param_node->fullname_with_scope() << " Get data type failed.";
    return nullptr;
  }

  ParameterPtr repeat_parameter;
  if (type_id == kNumberTypeFloat16) {
    std::vector<float16> repeat_data;
    auto data = static_cast<float16 *>(tensor->data_c());
    for (int r = 0; r < repeat_times; r++) {
      for (size_t i = 0; i < tensor->DataSize(); i++) {
        repeat_data.push_back(data[i]);
      }
    }
    MS_LOG(INFO) << "repeat_data size is" << repeat_data.size();
    repeat_parameter = opt::BuildFloat16VecParameterNode(
      func_graph, repeat_data, param_node->fullname_with_scope() + "_repeat_" + std::to_string(repeat_times));
  } else {
    std::vector<float> repeat_data;
    auto data = static_cast<float *>(tensor->data_c());
    for (int r = 0; r < repeat_times; r++) {
      for (size_t i = 0; i < tensor->DataSize(); i++) {
        repeat_data.push_back(data[i]);
      }
    }
    MS_LOG(INFO) << "repeat_data size is" << repeat_data.size();
    repeat_parameter = opt::BuildFloatVecParameterNode(
      func_graph, repeat_data, param_node->fullname_with_scope() + "_repeat_" + std::to_string(repeat_times));
  }

  auto abstract = repeat_parameter->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  ShapeVector shape_vector = {repeat_times, static_cast<int64_t>(tensor->DataSize())};
  abstract->set_shape(std::make_shared<abstract::Shape>(shape_vector));
  MS_LOG(INFO) << "After repeat parameter, repeat tensor shape is" << shape_vector;
  return repeat_parameter;
}

int FFNAntiquantFusion::GetExpertNumFromAntiQuantModeNodes(const AnfNodePtr &node) {
  CHECK_NULL_RETURN(node);
  if (!utils::isa<CNodePtr>(node) || !opt::CheckPrimitiveType(node, prim::kPrimMul)) {
    MS_LOG(ERROR) << "The node is not Mul node, fail to get expert num";
    return -1;
  }
  auto add_node = node->cast<CNodePtr>()->input(kInputIndexOne);
  if (!utils::isa<CNodePtr>(add_node) || !opt::CheckPrimitiveType(add_node, prim::kPrimAdd)) {
    MS_LOG(ERROR) << "The node is not Add node, fail to get expert num";
    return -1;
  }
  auto ascend_antiquant_node = add_node->cast<CNodePtr>()->input(kInputIndexOne);
  if (!utils::isa<CNodePtr>(ascend_antiquant_node) ||
      !opt::CheckPrimitiveType(ascend_antiquant_node, prim::kPrimAntiQuant)) {
    MS_LOG(ERROR) << "The node is not AscendAntiquant node, fail to get expert num";
    return -1;
  }
  auto weight_node = ascend_antiquant_node->cast<CNodePtr>()->input(kInputIndexOne);
  if (opt::CheckPrimitiveType(weight_node, prim::kPrimLoad)) {
    weight_node = weight_node->cast<CNodePtr>()->input(kInputIndexOne);
  }
  if (!weight_node->isa<Parameter>() || !weight_node->cast<ParameterPtr>()->has_default()) {
    MS_LOG(ERROR) << "weight_node: " << weight_node->fullname_with_scope()
                  << " is not ParameterPtr or have no default param";
    return -1;
  }
  ParameterPtr weight_param_node = weight_node->cast<ParameterPtr>();
  ShapeVector shape;
  if (opt::FetchShapeFromAbstract(weight_param_node->abstract(), &shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "fetch shape failed." << weight_param_node->fullname_with_scope();
    return -1;
  }
  if (shape.size() == 2) {
    MS_LOG(INFO) << "FFN weight shape size is 2 dims, shape is " << shape << ", get expert dim is 1";
    return 1;
  } else if (shape.size() != 3) {
    MS_LOG(ERROR) << "Not support weight shape is " << shape;
    return -1;
  }
  MS_LOG(INFO) << "MOE weight shape is " << shape << ", get expert dim is " << shape[0];
  return shape[0];
}

// weight in_strategy: MOE: [E, N, H], FFN: [N, H]
// scale or offset in_strategy: MOE: [E, H], FFN: [1, H]
std::vector<int64_t> FFNAntiquantFusion::GetScaleZpInStragety(std::vector<int64_t> weight_in_strategy) {
  if (weight_in_strategy.size() == 2) {
    return std::vector<int64_t>{1, weight_in_strategy[1]};
  } else if (weight_in_strategy.size() == 3) {
    return std::vector<int64_t>{weight_in_strategy[0], weight_in_strategy[2]};
  }
  MS_LOG(ERROR) << "Only support weight_in_strategy size is 2 or 3, but size is " << weight_in_strategy.size();
  return {};
}

int FFNAntiquantFusion::SetInStragety(const CNodePtr &ffn_cnode, const CNodePtr &ffn_fusion_cnode) {
  auto ffn_primitive = GetValueNode<std::shared_ptr<mindspore::Primitive>>(ffn_cnode->input(0));
  MS_EXCEPTION_IF_NULL(ffn_primitive);
  std::vector<std::vector<int64_t>> in_strategy;
  in_strategy = lite::quant::ExtractStrategy(ffn_primitive->GetAttr(IN_STRATEGY));
  MS_LOG(INFO) << "cnode: " << ffn_cnode->fullname_with_scope() << " in strategy is " << in_strategy;

  std::vector<int64_t> scale_zp_1_in_strategy = GetScaleZpInStragety(in_strategy[kFFNWeight1InputIndex]);
  CHECK_EQUAL_RETURN(scale_zp_1_in_strategy.size(), 0);
  std::vector<int64_t> scale_zp_2_in_strategy = GetScaleZpInStragety(in_strategy[kFFNWeight2InputIndex]);
  CHECK_EQUAL_RETURN(scale_zp_2_in_strategy.size(), 0);

  // set in strategy: antiquant_scale1, antiquant_scale2, antiquant_offset1, antiquant_offset2
  in_strategy.push_back(scale_zp_1_in_strategy);
  in_strategy.push_back(scale_zp_2_in_strategy);
  in_strategy.push_back(scale_zp_1_in_strategy);
  in_strategy.push_back(scale_zp_2_in_strategy);
  MS_LOG(INFO) << "ffn fusion cnode: " << ffn_fusion_cnode->fullname_with_scope() << " in strategy is " << in_strategy;

  auto ffn_fusion_primitive = GetValueNode<std::shared_ptr<mindspore::Primitive>>(ffn_fusion_cnode->input(0));
  CHECK_NULL_RETURN(ffn_fusion_primitive);
  ffn_fusion_primitive->AddAttr(IN_STRATEGY, MakeValue(in_strategy));
  return RET_OK;
}

CNodePtr FFNAntiquantFusion::NewFFNCNodeWithAntiquantFusion(const FuncGraphPtr &func_graph, const CNodePtr &ffn_cnode,
                                                            const ParameterPtr weight1_scale_node,
                                                            const ParameterPtr weight2_scale_node,
                                                            const ParameterPtr weight1_zp_node,
                                                            const ParameterPtr weight2_zp_node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(ffn_cnode);
  MS_EXCEPTION_IF_NULL(weight1_scale_node);
  MS_EXCEPTION_IF_NULL(weight2_scale_node);
  MS_EXCEPTION_IF_NULL(weight1_zp_node);
  MS_EXCEPTION_IF_NULL(weight2_zp_node);
  auto ffn_inputs = ffn_cnode->inputs();

  // Insert none value node, if option input is not set
  auto value = std::make_shared<None>();
  auto value_node = NewValueNode(value);
  MS_EXCEPTION_IF_NULL(value_node);
  value_node->set_abstract(std::make_shared<abstract::AbstractNone>());
  func_graph->NewCNode({value_node});
  for (size_t i = ffn_cnode->inputs().size(); i <= kFFNAntiquantScaleInputIndex; i++) {
    ffn_inputs.push_back(value_node);
  }

  ffn_inputs.push_back(weight1_scale_node);
  ffn_inputs.push_back(weight2_scale_node);
  ffn_inputs.push_back(weight1_zp_node);
  ffn_inputs.push_back(weight2_zp_node);
  auto ffn_fusion_cnode = func_graph->NewCNode(ffn_inputs);
  ffn_fusion_cnode->set_fullname_with_scope(ffn_cnode->fullname_with_scope() + "-AntiQuant-Fusion");
  ffn_fusion_cnode->set_abstract(ffn_cnode->abstract()->Clone());
  return ffn_fusion_cnode;
}

int FFNAntiquantFusion::Process(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  CHECK_NULL_RETURN(func_graph);
  CHECK_NULL_RETURN(cnode);

  auto weight1_node = cnode->input(kInputIndexTwo);
  auto weight2_node = cnode->input(kInputIndexThree);
  if (!lite::quant::IsAntiQuantModeNodes(weight1_node) || !lite::quant::IsAntiQuantModeNodes(weight2_node)) {
    MS_LOG(INFO) << "There is no antiquant node on the FFN node: " << cnode->fullname_with_scope();
    return lite::RET_OK;
  }

  // Get Scale & ZP Parameter
  ParameterPtr weight1_scale_node;
  ParameterPtr weight1_zp_node;
  auto ret = lite::quant::GetScaleZpFromAntiQuantModeNodes(weight1_node, &weight1_scale_node, &weight1_zp_node);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fail to GetScaleZpFromAntiQuantModeNodes: weight1 node: " << weight1_node->fullname_with_scope();
    return RET_ERROR;
  }
  ParameterPtr weight2_scale_node;
  ParameterPtr weight2_zp_node;
  ret = lite::quant::GetScaleZpFromAntiQuantModeNodes(weight2_node, &weight2_scale_node, &weight2_zp_node);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fail to GetScaleZpFromAntiQuantModeNodes: weight2 node: " << weight2_node->fullname_with_scope();
    return RET_ERROR;
  }

  // Repeat Quant Parameter
  int repeat_times_1 = GetExpertNumFromAntiQuantModeNodes(weight1_node);
  if (repeat_times_1 == -1) {
    MS_LOG(ERROR) << "Fail to GetExpertNumFromAntiQuantModeNodes : " << weight1_node->fullname_with_scope();
    return RET_ERROR;
  }
  int repeat_times_2 = GetExpertNumFromAntiQuantModeNodes(weight2_node);
  if (repeat_times_2 == -1) {
    MS_LOG(ERROR) << "Fail to GetExpertNumFromAntiQuantModeNodes : " << weight2_node->fullname_with_scope();
    return RET_ERROR;
  }
  auto weight1_scale_repeat_node = RepeatParameter(func_graph, weight1_scale_node, repeat_times_1);
  auto weight1_zp_repeat_node = RepeatParameter(func_graph, weight1_zp_node, repeat_times_1);
  auto weight2_scale_repeat_node = RepeatParameter(func_graph, weight2_scale_node, repeat_times_2);
  auto weight2_zp_repeat_node = RepeatParameter(func_graph, weight2_zp_node, repeat_times_2);

  // Remove weight1 & weight2 antiquant mode nodes (antiquant-> add -> mul)
  ret = lite::quant::RemoveAntiQuantModeNodes(func_graph, cnode, kIndex2);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fail to Remove weight1 AntiQuantMode Nodes";
    return RET_ERROR;
  }

  ret = lite::quant::RemoveAntiQuantModeNodes(func_graph, cnode, kIndex3);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fail to Remove weight2 AntiQuantMode Nodes";
    return RET_ERROR;
  }

  // Replace origin ffn nodes with ffn_fusion node
  auto ffn_fusion_cnode =
    NewFFNCNodeWithAntiquantFusion(func_graph, cnode, weight1_scale_repeat_node, weight2_scale_repeat_node,
                                   weight1_zp_repeat_node, weight2_zp_repeat_node);
  if (ffn_fusion_cnode == nullptr) {
    MS_LOG(ERROR) << "Fail to NewFFNCNodeWithAntiquantFusion";
    return RET_ERROR;
  }

  // Set Scale & ZP InStragety
  auto primitive = GetValueNode<std::shared_ptr<mindspore::Primitive>>(cnode->input(0));
  CHECK_NULL_RETURN(primitive);
  if (primitive->HasAttr(IN_STRATEGY)) {
    ret = SetInStragety(cnode, ffn_fusion_cnode);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Fail to SetInStragety.";
      return RET_ERROR;
    }
  }
  auto manager = Manage(func_graph);
  CHECK_NULL_RETURN(manager);
  (void)manager->Replace(cnode, ffn_fusion_cnode);
  MS_LOG(INFO) << "FFN Antiquant Fusion success, the cnode is " << ffn_fusion_cnode->fullname_with_scope();
  return RET_OK;
}
}  // namespace opt
}  // namespace mindspore
