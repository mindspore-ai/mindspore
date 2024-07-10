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
#include "tools/optimizer/fusion/flash_attention_antiquant_fusion.h"
#include <vector>
#include <memory>
#include "mindspore/core/ops/array_ops.h"
#include "ir/func_graph.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "ops/op_name.h"
#include "ops/other_ops.h"
#include "mindspore/lite/tools/optimizer/common/gllo_utils.h"
#include "mindspore/lite/tools/converter/quantizer/quantize_util.h"
#include "ops/ops_func_impl/incre_flash_attention.h"

namespace mindspore {
namespace {
constexpr size_t kFlashAttentionAntiquantScaleInputIndex = 11;
}  // namespace
namespace opt {
bool FlashAttentionAntiquantFusion::Run(const FuncGraphPtr &func_graph) {
  CHECK_NULL_RETURN(func_graph);
  auto node_list = TopoSort(func_graph->get_return());
  for (auto &node : node_list) {
    if (!utils::isa<CNode>(node) ||
        !CheckPrimitiveType(node, std::make_shared<Primitive>(ops::kNameIncreFlashAttention))) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    if (Process(func_graph, cnode) != lite::RET_OK) {
      MS_LOG(ERROR) << "Do FlashAttentionAntiquantFusion failed, node name is " << node->fullname_with_scope();
      return false;
    }
  }
  return true;
}

CNodePtr FlashAttentionAntiquantFusion::NewFlashAttentionCNodeWithAntiquantFusion(const FuncGraphPtr &func_graph,
                                                                                  const CNodePtr &fa_cnode,
                                                                                  const ParameterPtr scale,
                                                                                  const ParameterPtr offset) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(fa_cnode);
  MS_EXCEPTION_IF_NULL(scale);
  MS_EXCEPTION_IF_NULL(offset);
  auto fa_inputs = fa_cnode->inputs();
  MS_LOG(INFO) << fa_cnode->fullname_with_scope() << " fa_inputs " << fa_inputs.size();

  // Insert none value node, if option input is not set
  auto value = std::make_shared<None>();
  auto value_node = NewValueNode(value);
  MS_EXCEPTION_IF_NULL(value_node);
  value_node->set_abstract(std::make_shared<abstract::AbstractNone>());
  func_graph->NewCNode({value_node});
  for (size_t i = fa_cnode->inputs().size(); i <= kFlashAttentionAntiquantScaleInputIndex; i++) {
    MS_LOG(INFO) << fa_cnode->fullname_with_scope() << " index " << i << " is AbstractNone";
    fa_inputs.push_back(value_node);
  }
  if (fa_inputs.size() <= kFlashAttentionAntiquantScaleInputIndex + 1) {
    fa_inputs.push_back(scale);
  } else {
    fa_inputs.at(kFlashAttentionAntiquantScaleInputIndex + 1) = scale;
  }
  if (fa_inputs.size() <= kFlashAttentionAntiquantScaleInputIndex + 2) {
    fa_inputs.push_back(offset);
  } else {
    fa_inputs.at(kFlashAttentionAntiquantScaleInputIndex + 2) = offset;
  }
  auto fa_fusion_cnode = func_graph->NewCNode(fa_inputs);
  fa_fusion_cnode->set_fullname_with_scope(fa_cnode->fullname_with_scope() + "-AntiQuant-Fusion");
  fa_fusion_cnode->set_abstract(fa_cnode->abstract()->Clone());
  return fa_fusion_cnode;
}

// Concat two param_node
// param_node_1 shape: [1, N...]
// param_node_2 shape: [1, N...]
// concat param_node shape: [2, N...]
ParameterPtr FlashAttentionAntiquantFusion::ConcatParameter(const FuncGraphPtr &func_graph,
                                                            const ParameterPtr param_node_1,
                                                            const ParameterPtr param_node_2, std::string name) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(param_node_1);
  MS_EXCEPTION_IF_NULL(param_node_2);
  if (!(param_node_1)->has_default() || !(param_node_2)->has_default()) {
    MS_LOG(ERROR) << "param_node: " << param_node_1->fullname_with_scope() << " or "
                  << param_node_2->fullname_with_scope() << " don't have default param";
    return nullptr;
  }
  ShapeVector ori_shape;
  if (opt::FetchShapeFromAbstract(param_node_1->abstract(), &ori_shape) != lite::RET_OK) {
    MS_LOG(ERROR) << "fetch shape failed." << param_node_1->fullname_with_scope();
    return nullptr;
  }
  if (ori_shape[0] != 1) {
    MS_LOG(ERROR) << "Only support param shape[0] is 1, but the param shape is " << ori_shape;
    return nullptr;
  }

  auto tensor_1 = std::static_pointer_cast<tensor::Tensor>(param_node_1->default_param());
  auto tensor_2 = std::static_pointer_cast<tensor::Tensor>(param_node_2->default_param());
  if (tensor_1 == nullptr || tensor_2 == nullptr) {
    MS_LOG(ERROR) << "default_param can not cast to tensor::Tensor";
    return nullptr;
  }
  if (tensor_1->Size() != tensor_2->Size()) {
    MS_LOG(ERROR) << "tensor_1 size: " << tensor_1->Size() << " but tensor_2 size: " << tensor_2->Size()
                  << ", they are not same.";
    return nullptr;
  }
  TypeId type_id;
  if (opt::GetDataTypeFromAnfNode(param_node_1, &type_id) != RET_OK) {
    MS_LOG(WARNING) << param_node_1->fullname_with_scope() << " Get data type failed.";
    return nullptr;
  }

  ParameterPtr concat_parameter;
  if (type_id == kNumberTypeFloat16) {
    std::vector<float16> concat_data;
    auto key_data = static_cast<float16 *>(tensor_1->data_c());
    auto value_data = static_cast<float16 *>(tensor_2->data_c());
    for (size_t i = 0; i < tensor_1->DataSize(); i++) {
      concat_data.push_back(key_data[i]);
    }
    for (size_t i = 0; i < tensor_2->DataSize(); i++) {
      concat_data.push_back(value_data[i]);
    }
    MS_LOG(INFO) << "concat_data size is" << concat_data.size();
    concat_parameter = opt::BuildFloat16VecParameterNode(func_graph, concat_data, name);
  } else {
    std::vector<float> concat_data;
    auto key_data = static_cast<float *>(tensor_1->data_c());
    auto value_data = static_cast<float *>(tensor_2->data_c());
    for (size_t i = 0; i < tensor_1->DataSize(); i++) {
      concat_data.push_back(key_data[i]);
    }
    for (size_t i = 0; i < tensor_2->DataSize(); i++) {
      concat_data.push_back(value_data[i]);
    }
    MS_LOG(INFO) << "concat_data size is" << concat_data.size();
    concat_parameter = opt::BuildFloatVecParameterNode(func_graph, concat_data, name);
  }

  auto abstract = concat_parameter->abstract();
  MS_EXCEPTION_IF_NULL(abstract);
  ShapeVector shape_vector = {2};
  shape_vector.insert(shape_vector.end(), ori_shape.begin() + 1, ori_shape.end());
  abstract->set_shape(std::make_shared<abstract::Shape>(shape_vector));
  MS_LOG(INFO) << "After concat parameter, concat tensor shape is" << shape_vector;
  return concat_parameter;
}

int FlashAttentionAntiquantFusion::Process(const FuncGraphPtr &func_graph, const CNodePtr &cnode) {
  CHECK_NULL_RETURN(func_graph);
  CHECK_NULL_RETURN(cnode);

  auto weight1_node = cnode->input(kInputIndexTwo);
  auto weight2_node = cnode->input(kInputIndexThree);
  if (!lite::quant::IsAntiQuantModeNodes(weight1_node)) {
    MS_LOG(INFO) << "There is no antiquant node on the IncreFlashAttention node: "
                 << weight1_node->fullname_with_scope();
    return lite::RET_OK;
  }
  if (!lite::quant::IsAntiQuantModeNodes(weight2_node)) {
    MS_LOG(INFO) << "There is no antiquant node on the IncreFlashAttention node: "
                 << weight2_node->fullname_with_scope();
    return lite::RET_OK;
  }

  // Get Concat Scale & ZP
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
  ParameterPtr concat_scale =
    ConcatParameter(func_graph, weight1_scale_node, weight2_scale_node, cnode->fullname_with_scope() + "_concat_scale");
  ParameterPtr concat_zp =
    ConcatParameter(func_graph, weight1_zp_node, weight2_zp_node, cnode->fullname_with_scope() + "_concat_zp");

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

  // Replace origin fa nodes with fa_fusion node
  auto fa_fusion_cnode = NewFlashAttentionCNodeWithAntiquantFusion(func_graph, cnode, concat_scale, concat_zp);
  if (fa_fusion_cnode == nullptr) {
    MS_LOG(ERROR) << "Fail to NewFlashAttentionCNodeWithAntiquantFusion";
    return RET_ERROR;
  }
  auto manager = Manage(func_graph);
  CHECK_NULL_RETURN(manager);
  (void)manager->Replace(cnode, fa_fusion_cnode);
  MS_LOG(INFO) << "FlashAttention Antiquant Fusion success, the cnode is " << fa_fusion_cnode->fullname_with_scope();

  return RET_OK;
}
}  // namespace opt
}  // namespace mindspore
