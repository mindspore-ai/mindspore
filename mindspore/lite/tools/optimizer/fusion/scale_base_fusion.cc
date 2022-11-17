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
#include "tools/optimizer/fusion/scale_base_fusion.h"
#include <memory>
#include "tools/common/tensor_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::opt {
int ScaleBaseFusion::CalNewCnodeScale(const CNodePtr &curr_cnode,
                                      const std::vector<AnfNodePtr> &fusion_cnode_inputs) const {
  auto curr_weight_node = curr_cnode->input(kInputIndexTwo);
  std::shared_ptr<tensor::Tensor> curr_weight_tensor = GetTensorInfo(curr_weight_node);
  MS_CHECK_TRUE_RET(curr_weight_tensor != nullptr, RET_ERROR);
  if (curr_weight_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int64_t> curr_weight_shape = curr_weight_tensor->shape();
  auto curr_weight_data = reinterpret_cast<float *>(curr_weight_tensor->data_c());
  MS_CHECK_TRUE_RET(curr_weight_data != nullptr, RET_ERROR);

  auto fusion_weight_node = fusion_cnode_inputs[kInputIndexTwo];
  std::shared_ptr<tensor::Tensor> fusion_weight_tensor = GetTensorInfo(fusion_weight_node);
  if (fusion_weight_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int64_t> fusion_weight_shape = fusion_weight_tensor->shape();
  MS_CHECK_TRUE_RET(fusion_weight_shape.size() > 1, RET_ERROR);
  MS_CHECK_TRUE_RET(fusion_weight_shape[fusion_weight_shape.size() - 1] == curr_weight_shape[0], RET_ERROR);
  auto fusion_weight_data = reinterpret_cast<float *>(fusion_weight_tensor->data_c());
  MS_CHECK_TRUE_RET(fusion_weight_data != nullptr, RET_ERROR);

  return CalNewScaleImpl(curr_weight_data, fusion_weight_shape, fusion_weight_data, fusion_cnode_inputs[0]);
}

int ScaleBaseFusion::CalNewCnodeBias(const FuncGraphPtr &func_graph, const CNodePtr &curr_cnode,
                                     const std::vector<AnfNodePtr> &fusion_cnode_inputs) const {
  auto curr_weight_node = curr_cnode->input(kInputIndexTwo);
  std::shared_ptr<tensor::Tensor> curr_weight_tensor = GetTensorInfo(curr_weight_node);
  MS_CHECK_TRUE_MSG(curr_weight_tensor != nullptr, RET_ERROR, "node's weight is invalid,please check your model file");
  if (curr_weight_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int64_t> curr_weight_shape = curr_weight_tensor->shape();
  auto curr_weight_data = reinterpret_cast<float *>(curr_weight_tensor->data_c());
  MS_CHECK_TRUE_RET(curr_weight_data != nullptr, RET_ERROR);

  float *curr_bias_data = nullptr;
  if (curr_cnode->size() >= kInputSizeFour) {
    auto curr_bias_node = curr_cnode->input(kInputIndexThree);
    auto curr_bias_tensor = GetTensorInfo(curr_bias_node);
    if (curr_bias_tensor->data_type() != kNumberTypeFloat32) {
      MS_LOG(ERROR) << "only support float32 data type";
      return RET_ERROR;
    }
    std::vector<int64_t> curr_bias_shape = curr_bias_tensor->shape();
    MS_CHECK_TRUE_RET(curr_bias_shape[0] == curr_weight_shape[0], RET_ERROR);
    curr_bias_data = reinterpret_cast<float *>(curr_bias_tensor->data_c());
    MS_CHECK_TRUE_RET(curr_bias_data != nullptr, RET_ERROR);
  }

  auto fusion_bias_node = fusion_cnode_inputs[kInputIndexThree];
  std::shared_ptr<tensor::Tensor> fusion_bias_tensor = GetTensorInfo(fusion_bias_node);
  if (fusion_bias_tensor->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "only support float32 data type";
    return RET_ERROR;
  }
  std::vector<int64_t> fusion_bias_shape = fusion_bias_tensor->shape();
  MS_CHECK_TRUE_RET(fusion_bias_shape.size() == 1, RET_ERROR);
  float *fusion_bias_data = reinterpret_cast<float *>(fusion_bias_tensor->data_c());
  MS_CHECK_TRUE_RET(fusion_bias_data != nullptr, RET_ERROR);

  return CalNewBiasImpl(curr_weight_data, curr_bias_data, fusion_bias_shape, fusion_bias_data);
}

bool ScaleBaseFusion::CheckCurrCnodeProper(const CNodePtr &scale_cnode) const {
  if (!CheckPrimitiveType(scale_cnode, prim::kPrimScaleFusion)) {
    MS_LOG(INFO) << scale_cnode->fullname_with_scope() << "is not scale node";
    return false;
  }
  auto scale_weight_node = scale_cnode->input(kInputIndexTwo);
  if (!IsParamNode(scale_weight_node)) {
    MS_LOG(INFO) << scale_cnode->fullname_with_scope() << "'s weight is not parameter";
    return false;
  }
  if (scale_cnode->size() > kInputSizeThree) {
    auto scale_bias_node = scale_cnode->input(kInputIndexThree);
    if (!IsParamNode(scale_bias_node)) {
      MS_LOG(INFO) << scale_cnode->fullname_with_scope() << "'s bias is not parameter";
      return false;
    }
  }

  auto curr_primc = GetValueNode<PrimitiveCPtr>(scale_cnode->input(0));  // previous fc primitive
  MS_CHECK_TRUE_RET(curr_primc != nullptr, false);
  if (IsQuantParameterNode(curr_primc)) {
    MS_LOG(INFO) << scale_cnode->fullname_with_scope() << "is quant node";
    return false;
  }

  auto scale_prim = ops::GetOperator<ops::ScaleFusion>(scale_cnode->input(0));
  MS_CHECK_TRUE_RET(scale_prim != nullptr, false);
  auto axis_attr = scale_prim->GetAttr(ops::kAxis);
  MS_CHECK_TRUE_RET(axis_attr != nullptr, false);
  tensor::TensorPtr tensor_info;
  if (GetTensorInfoFromAbstract(&tensor_info, scale_cnode, 1) != RET_OK) {
    MS_LOG(ERROR) << "failed to create tensor";
    return false;
  }
  auto in_shape = tensor_info->shape();
  if (in_shape.size() <= 0) {
    MS_LOG(INFO) << "scale fusion not support dynamic dims";
    return false;
  }
  int64_t axis = scale_prim->get_axis() < 0 ? scale_prim->get_axis() + in_shape.size() : scale_prim->get_axis();
  if (axis != SizeToLong(in_shape.size() - 1)) {
    MS_LOG(INFO) << "scale axis must be the last dim of input";
    return false;
  }

  auto curr_weight_node = scale_cnode->input(kInputIndexTwo);
  std::shared_ptr<tensor::Tensor> curr_weight_tensor = GetTensorInfo(curr_weight_node);
  MS_CHECK_TRUE_RET(curr_weight_tensor != nullptr, false);
  std::vector<int64_t> curr_weight_shape = curr_weight_tensor->shape();
  std::shared_ptr<tensor::Tensor> curr_bias_tensor = GetTensorInfo(curr_weight_node);
  MS_CHECK_TRUE_RET(curr_bias_tensor != nullptr, false);
  std::vector<int64_t> curr_bias_shape = curr_bias_tensor->shape();
  if (curr_weight_shape != curr_bias_shape) {
    MS_LOG(INFO) << "scale fusion only support 1 dims";
    return false;
  }

  return true;
}

std::vector<AnfNodePtr> ScaleBaseFusion::GetNewCnodeInputs(const FuncGraphPtr &func_graph, const CNodePtr &curr_cnode,
                                                           const CNodePtr &prev_cnode) const {
  auto prim = BuildNewPrimitive(curr_cnode, prev_cnode);
  MS_CHECK_TRUE_MSG(prim != nullptr, {}, "failed to create primitive");
  auto prev_primc = NewValueNode(std::shared_ptr<ops::PrimitiveC>(prim));
  MS_CHECK_TRUE_MSG(prev_primc != nullptr, {}, "failed to create ValueNode");
  if (curr_cnode->size() == kInputSizeThree && prev_cnode->size() == kInputSizeThree) {
    return std::vector<AnfNodePtr>{prev_primc, prev_cnode->input(1), prev_cnode->input(kInputIndexTwo)};
  }
  if (prev_cnode->size() == kInputSizeThree) {
    auto curr_bias_node = curr_cnode->input(kInputIndexThree);
    auto curr_bias_tensor = GetTensorInfo(curr_bias_node);
    MS_CHECK_TRUE_RET(curr_bias_tensor != nullptr, {});
    std::vector<int64_t> curr_bias_shape = curr_bias_tensor->shape();
    MS_CHECK_TRUE_RET(curr_bias_shape.size() == 1, {});

    auto new_bias_node = func_graph->add_parameter();
    auto new_bias_shape = std::vector<int64_t>{curr_bias_shape[0]};
    auto tensor_info = lite::CreateTensorInfo(nullptr, 0, new_bias_shape, curr_bias_tensor->data_type());
    MS_CHECK_TRUE_RET(tensor_info != nullptr, {});
    if (EOK != memset_s(static_cast<float *>(tensor_info->data_c()), new_bias_shape[0] * sizeof(float), 0,
                        new_bias_shape[0] * sizeof(float))) {
      MS_LOG(ERROR) << "memset_s failed";
      return {};
    }

    auto status = lite::InitParameterFromTensorInfo(new_bias_node, tensor_info);
    MS_CHECK_TRUE_RET(status != RET_OK, {});
    auto bias_abstr = curr_cnode->input(kInputIndexThree)->abstract();
    MS_CHECK_TRUE_RET(bias_abstr != nullptr, {});
    new_bias_node->set_abstract(bias_abstr->Clone());
    new_bias_node->set_name(prev_cnode->fullname_with_scope() + "fusion_bias");
    return std::vector<AnfNodePtr>{prev_primc, prev_cnode->input(1), prev_cnode->input(kInputIndexTwo), new_bias_node};
  }
  return std::vector<AnfNodePtr>{prev_primc, prev_cnode->input(1), prev_cnode->input(kInputIndexTwo),
                                 prev_cnode->input(kInputIndexThree)};
}

const AnfNodePtr ScaleBaseFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                          const EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr) {
    return nullptr;
  }

  auto curr_cnode = node->cast<CNodePtr>();
  if (curr_cnode == nullptr || curr_cnode->size() < kInputSizeThree) {
    return nullptr;
  }
  auto prev_node = curr_cnode->input(1);
  auto prev_cnode = prev_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(prev_cnode != nullptr, nullptr);
  if (IsMarkedTrainOp(curr_cnode) || IsMarkedTrainOp(prev_cnode)) {
    return nullptr;
  }
  if (IsMultiOutputTensors(func_graph, prev_cnode)) {
    return nullptr;
  }

  if (!CheckCurrCnodeProper(curr_cnode) || !CheckPrevCnodeProper(prev_cnode)) {
    return nullptr;
  }
  auto fusion_cnode_inputs = GetNewCnodeInputs(func_graph, curr_cnode, prev_cnode);
  MS_CHECK_TRUE_RET(fusion_cnode_inputs.size() > 0, nullptr);
  if (fusion_cnode_inputs.size() == kInputSizeFour) {
    auto ret = CalNewCnodeBias(func_graph, curr_cnode, fusion_cnode_inputs);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << curr_cnode->fullname_with_scope() << " failed to fusion bias with "
                    << prev_cnode->fullname_with_scope();
      return nullptr;
    }
  }
  if (CalNewCnodeScale(curr_cnode, fusion_cnode_inputs) != RET_OK) {
    MS_LOG(ERROR) << curr_cnode->fullname_with_scope() << " failed to fusion with "
                  << prev_cnode->fullname_with_scope();
    return nullptr;
  }

  auto fusion_cnode = func_graph->NewCNode(fusion_cnode_inputs);
  auto manager = func_graph->manager();
  for (size_t i = 0; i < fusion_cnode->inputs().size(); ++i) {
    manager->SetEdge(fusion_cnode, i, fusion_cnode_inputs[i]);
  }
  MS_CHECK_TRUE_RET(fusion_cnode != nullptr, nullptr);
  fusion_cnode->set_fullname_with_scope(prev_cnode->fullname_with_scope() + "scale");
  MS_CHECK_TRUE_RET(prev_cnode->abstract() != nullptr, nullptr);
  fusion_cnode->set_abstract(prev_cnode->abstract()->Clone());
  MS_LOG(INFO) << curr_cnode->fullname_with_scope() << " fusion success";
  return fusion_cnode;
}
}  // namespace mindspore::opt
