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
#include "tools/optimizer/fusion/scale_scale_fusion.h"
#include <functional>
#include <memory>
#include "tools/converter/quantizer/quant_param_holder.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/common/tensor_util.h"
#include "ops/fusion/scale_fusion.h"
#include "securec/include/securec.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::opt {
namespace {
constexpr size_t kScaleWeightIndex = 2;
constexpr size_t kScaleBiasIndex = 3;
constexpr size_t kScaleNoBiasLen = 3;
constexpr size_t kScaleWithBiasLen = 4;
}  // namespace

const BaseRef ScaleScaleFusion::DefinePattern() const {
  auto is_scale_up = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimScaleFusion>);
  MS_CHECK_TRUE_RET(is_scale_up != nullptr, {});
  auto is_param = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(is_param != nullptr, {});
  auto is_seq_var = std::make_shared<SeqVar>();
  MS_CHECK_TRUE_RET(is_seq_var != nullptr, {});
  auto is_scale_down = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimScaleFusion>);
  MS_CHECK_TRUE_RET(is_scale_down != nullptr, {});
  return VectorRef({is_scale_down, is_scale_up, is_param, is_seq_var});
}

bool ScaleScaleFusion::CheckScaleNode(const CNodePtr &scale_cnode) const {
  MS_ASSERT(scale_cnode != nullptr);
  if (IsMarkedTrainOp(scale_cnode)) {
    return false;
  }
  MS_CHECK_TRUE_RET(scale_cnode->size() >= kScaleNoBiasLen, false);
  auto scale_prim = ops::GetOperator<ops::ScaleFusion>(scale_cnode->input(FIRST_INPUT));
  MS_CHECK_TRUE_RET(scale_prim != nullptr, false);
  auto scale_prim_c = scale_prim->GetPrim();
  MS_CHECK_TRUE_RET(scale_prim_c != nullptr, false);
  auto quant_attr = scale_prim_c->GetAttr("quant_params");
  if (quant_attr != nullptr) {
    auto quant_param_holder = quant_attr->cast<lite::QuantParamHolderPtr>();
    MS_CHECK_TRUE_RET(quant_param_holder != nullptr, false);
    auto quant_params = quant_param_holder->get_input_quant_params();
    bool is_quant = std::any_of(quant_params.begin(), quant_params.end(), [](std::vector<schema::QuantParamT> &params) {
      return !params.empty() && params.front().inited;
    });
    if (is_quant) {
      return false;
    }
  }

  auto scale_weight_node = scale_cnode->input(kScaleWeightIndex);
  if (!IsParamNode(scale_weight_node)) {
    return false;
  }
  if (scale_cnode->size() == kScaleWithBiasLen) {
    auto scale_bias_node = scale_cnode->input(kScaleWeightIndex);
    MS_CHECK_TRUE_RET(scale_bias_node != nullptr, false);
    if (!IsParamNode(scale_bias_node)) {
      return false;
    }
  }
  return true;
}

int ScaleScaleFusion::GetInputParamsAndTensors(const CNodePtr &up_scale_cnode, const CNodePtr &down_scale_cnode) const {
  MS_ASSERT(up_scale_cnode != nullptr && down_scale_cnode != nullptr);
  auto abstract = GetCNodeInputAbstract(up_scale_cnode, SECOND_INPUT);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Get abstract failed.";
    return lite::RET_ERROR;
  }
  if (FetchShapeFromAbstract(abstract, &scale_input_shape_) != lite::RET_OK) {
    MS_LOG(ERROR) << "Fetch shape from abstract failed.";
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_RET(!scale_input_shape_.empty(), lite::RET_ERROR);

  auto up_scale_prim = ops::GetOperator<ops::ScaleFusion>(up_scale_cnode->input(FIRST_INPUT));
  MS_CHECK_TRUE_RET(up_scale_prim != nullptr, lite::RET_ERROR);
  auto up_scale_prim_c = up_scale_prim->GetPrim();
  MS_CHECK_TRUE_RET(up_scale_prim_c != nullptr && up_scale_prim_c->GetAttr(ops::kAxis), lite::RET_ERROR);
  auto axis = up_scale_prim->get_axis();
  up_scale_axis_ = axis < 0 ? axis + static_cast<int>(scale_input_shape_.size()) : axis;
  auto down_scale_prim = ops::GetOperator<ops::ScaleFusion>(down_scale_cnode->input(FIRST_INPUT));
  MS_CHECK_TRUE_RET(down_scale_prim != nullptr, lite::RET_ERROR);
  auto down_scale_prim_c = down_scale_prim->GetPrim();
  MS_CHECK_TRUE_RET(down_scale_prim_c != nullptr && down_scale_prim_c->GetAttr(ops::kAxis), lite::RET_ERROR);
  axis = down_scale_prim->get_axis();
  down_scale_axis_ = axis < 0 ? axis + static_cast<int>(scale_input_shape_.size()) : axis;

  auto up_weight_param = up_scale_cnode->input(THIRD_INPUT);
  MS_CHECK_TRUE_RET(up_weight_param != nullptr, lite::RET_ERROR);
  up_weight_tensor_ = GetTensorInfo(up_weight_param);
  MS_CHECK_TRUE_RET(up_weight_tensor_ != nullptr, lite::RET_ERROR);
  MS_CHECK_TRUE_RET(
    up_weight_tensor_->data_type() == kNumberTypeFloat || up_weight_tensor_->data_type() == kNumberTypeFloat32,
    lite::RET_ERROR);
  if (up_scale_cnode->size() == kScaleWithBiasLen) {
    auto up_bias_param = up_scale_cnode->input(FOURTH_INPUT);
    MS_CHECK_TRUE_RET(up_bias_param != nullptr, lite::RET_ERROR);
    up_bias_tensor_ = GetTensorInfo(up_bias_param);
    MS_CHECK_TRUE_RET(up_bias_tensor_ != nullptr, lite::RET_ERROR);
    MS_CHECK_TRUE_RET(
      up_bias_tensor_->data_type() == kNumberTypeFloat || up_bias_tensor_->data_type() == kNumberTypeFloat32,
      lite::RET_ERROR);
  }

  auto down_weight_param = down_scale_cnode->input(THIRD_INPUT);
  MS_CHECK_TRUE_RET(down_weight_param != nullptr, lite::RET_ERROR);
  down_weight_tensor_ = GetTensorInfo(down_weight_param);
  MS_CHECK_TRUE_RET(down_weight_tensor_ != nullptr, lite::RET_ERROR);
  MS_CHECK_TRUE_RET(
    down_weight_tensor_->data_type() == kNumberTypeFloat || down_weight_tensor_->data_type() == kNumberTypeFloat32,
    lite::RET_ERROR);
  if (down_scale_cnode->size() == kScaleWithBiasLen) {
    auto down_bias_param = down_scale_cnode->input(FOURTH_INPUT);
    MS_CHECK_TRUE_RET(down_bias_param != nullptr, lite::RET_ERROR);
    down_bias_tensor_ = GetTensorInfo(down_bias_param);
    MS_CHECK_TRUE_RET(down_bias_tensor_ != nullptr, lite::RET_ERROR);
    MS_CHECK_TRUE_RET(
      down_bias_tensor_->data_type() == kNumberTypeFloat || down_bias_tensor_->data_type() == kNumberTypeFloat32,
      lite::RET_ERROR);
  }
  return lite::RET_OK;
}

tensor::TensorPtr ScaleScaleFusion::GetMultiplyResultTensorInfo(const tensor::TensorPtr &left_tensor,
                                                                const tensor::TensorPtr &right_tensor) const {
  MS_ASSERT(left_tensor != nullptr && right_tensor != nullptr);
  auto left_weight_shape = left_tensor->shape();
  auto right_weight_shape = right_tensor->shape();
  size_t left_end_idx = up_scale_axis_ + left_weight_shape.size();
  size_t right_end_idx = down_scale_axis_ + right_weight_shape.size();
  auto begin_idx = MSMIN(up_scale_axis_, down_scale_axis_);
  auto tmp_idx = MSMAX(up_scale_axis_, down_scale_axis_);
  auto tmp_end_idx = up_scale_axis_ < down_scale_axis_ ? right_end_idx : left_end_idx;
  size_t expand_size = 1;
  for (size_t i = begin_idx; i < tmp_idx; i++) {
    MS_CHECK_TRUE_RET(!SIZE_MUL_OVERFLOW(expand_size, static_cast<size_t>(scale_input_shape_.at(i))), nullptr);
    expand_size *= static_cast<size_t>(scale_input_shape_.at(i));
  }
  size_t ele_size = 1;
  for (size_t i = tmp_idx; i < tmp_end_idx; i++) {
    MS_CHECK_TRUE_RET(!SIZE_MUL_OVERFLOW(ele_size, static_cast<size_t>(scale_input_shape_.at(i))), nullptr);
    ele_size *= static_cast<size_t>(scale_input_shape_.at(i));
  }
  MS_CHECK_LE(expand_size * ele_size * sizeof(float), MAX_MALLOC_SIZE, nullptr);
  float *expand_data = reinterpret_cast<float *>(malloc(expand_size * ele_size * sizeof(float)));
  if (expand_data == nullptr) {
    MS_LOG(ERROR) << "malloc data failed.";
    return nullptr;
  }
  auto tmp_tensor = up_scale_axis_ < down_scale_axis_ ? right_tensor : left_tensor;
  for (size_t i = 0; i < expand_size; i++) {
    if (memcpy_s(expand_data + i * ele_size, ele_size * sizeof(float), tmp_tensor->data_c(), tmp_tensor->Size()) !=
        EOK) {
      MS_LOG(ERROR) << "memcpy data failed.";
      free(expand_data);
      return nullptr;
    }
  }

  float *left_data = nullptr;
  float *right_data = nullptr;
  if (up_scale_axis_ < down_scale_axis_) {
    left_data = left_end_idx < right_end_idx ? static_cast<float *>(left_tensor->data_c()) : expand_data;
    right_data = left_end_idx < right_end_idx ? expand_data : static_cast<float *>(left_tensor->data_c());
  } else {
    left_data = left_end_idx < right_end_idx ? expand_data : static_cast<float *>(right_tensor->data_c());
    right_data = left_end_idx < right_end_idx ? static_cast<float *>(right_tensor->data_c()) : expand_data;
  }
  if (left_data == nullptr || right_data == nullptr) {
    free(expand_data);
    return nullptr;
  }

  auto end_idx = MSMAX(left_end_idx, right_end_idx);
  expand_shape_.assign(scale_input_shape_.begin() + begin_idx, scale_input_shape_.begin() + end_idx);
  auto tensor_info = lite::CreateTensorInfo(nullptr, 0, expand_shape_, left_tensor->data_type());
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed.";
    free(expand_data);
    return nullptr;
  }
  float *new_weight_data = reinterpret_cast<float *>(tensor_info->data_c());
  MS_ASSERT(new_weight_data != nullptr);
  size_t outer_size =
    std::accumulate(scale_input_shape_.begin() + begin_idx,
                    scale_input_shape_.begin() + MSMIN(left_end_idx, right_end_idx), 1, std::multiplies<size_t>());
  size_t inner_size = std::accumulate(scale_input_shape_.begin() + MSMIN(left_end_idx, right_end_idx),
                                      scale_input_shape_.begin() + end_idx, 1, std::multiplies<size_t>());
  for (size_t i = 0; i < outer_size; i++) {
    for (size_t j = 0; j < inner_size; j++) {
      new_weight_data[i * inner_size + j] = left_data[i] * right_data[i * inner_size + j];
    }
  }
  free(expand_data);
  return tensor_info;
}

ParameterPtr ScaleScaleFusion::GenerateNewWeightNode(const FuncGraphPtr &func_graph, const std::string &name) const {
  auto param = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param != nullptr, nullptr);
  auto new_weight_tensor = GetMultiplyResultTensorInfo(up_weight_tensor_, down_weight_tensor_);
  if (new_weight_tensor == nullptr) {
    MS_LOG(ERROR) << "Get new weight tensor failed.";
    return nullptr;
  }
  if (lite::InitParameterFromTensorInfo(param, new_weight_tensor) != lite::RET_OK) {
    MS_LOG(ERROR) << "Init parameter from tensor info failed.";
    return nullptr;
  }
  param->set_name(name);
  return param;
}

ParameterPtr ScaleScaleFusion::GenerateNewBiasNode(const FuncGraphPtr &func_graph, const std::string &name) const {
  auto param = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(param != nullptr, nullptr);
  tensor::TensorPtr tensor_info = GetMultiplyResultTensorInfo(up_bias_tensor_, down_weight_tensor_);
  if (tensor_info == nullptr) {
    MS_LOG(ERROR) << "Create tensor info failed.";
    return nullptr;
  }
  if (down_bias_tensor_ != nullptr) {
    auto bias_shape = down_bias_tensor_->shape_c();
    int axis_diff = down_scale_axis_ - MSMIN(up_scale_axis_, down_scale_axis_);
    int end_idx_diff =
      down_scale_axis_ + static_cast<int>(bias_shape.size()) -
      MSMAX(down_scale_axis_ + bias_shape.size(), up_scale_axis_ + up_weight_tensor_->shape_c().size());
    size_t outer_size = axis_diff > 0 ? std::accumulate(expand_shape_.begin(), expand_shape_.begin() + axis_diff, 1,
                                                        std::multiplies<size_t>())
                                      : 1;
    size_t inner_size = end_idx_diff < 0 ? std::accumulate(expand_shape_.end() + end_idx_diff, expand_shape_.end(), 1,
                                                           std::multiplies<size_t>())
                                         : 1;
    size_t bias_size = std::accumulate(bias_shape.begin(), bias_shape.end(), 1, std::multiplies<size_t>());
    float *bias_data = reinterpret_cast<float *>(down_bias_tensor_->data_c());
    float *data = reinterpret_cast<float *>(tensor_info->data_c());
    MS_ASSERT(bias_data != nullptr && data != nullptr);
    for (size_t i = 0; i < outer_size; i++) {
      for (size_t j = 0; j < bias_size; j++) {
        for (size_t k = 0; k < inner_size; k++) {
          data[i * bias_size * inner_size + j * inner_size + k] += bias_data[j];
        }
      }
    }
  }
  if (lite::InitParameterFromTensorInfo(param, tensor_info) != lite::RET_OK) {
    MS_LOG(ERROR) << "Init parameter from tensor info failed.";
    return nullptr;
  }
  param->set_name(name);
  return param;
}

const AnfNodePtr ScaleScaleFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                           const EquivPtr &) const {
  MS_ASSERT(func_graph != nullptr && node != nullptr);
  auto down_scale_cnode = node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(down_scale_cnode != nullptr, nullptr);
  auto up_scale_node = down_scale_cnode->input(SECOND_INPUT);
  MS_CHECK_TRUE_RET(up_scale_node != nullptr, nullptr);
  auto up_scale_cnode = up_scale_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(up_scale_cnode != nullptr, nullptr);
  if (IsMultiOutputTensors(func_graph, up_scale_cnode)) {
    return nullptr;
  }
  if (!CheckScaleNode(up_scale_cnode) || !CheckScaleNode(down_scale_cnode)) {
    return nullptr;
  }
  auto scale_prim = ops::GetOperator<ops::ScaleFusion>(up_scale_cnode->input(FIRST_INPUT));
  MS_CHECK_TRUE_RET(scale_prim != nullptr, nullptr);
  auto scale_prim_c = scale_prim->GetPrim();
  MS_CHECK_TRUE_RET(scale_prim_c != nullptr, nullptr);
  if (scale_prim_c->GetAttr(ops::kActivationType) != nullptr && scale_prim->get_activation_type() != NO_ACTIVATION) {
    return nullptr;
  }

  if (GetInputParamsAndTensors(up_scale_cnode, down_scale_cnode) != lite::RET_OK) {
    MS_LOG(ERROR) << "Get inputs failed.";
    return nullptr;
  }
  auto new_weight_param = GenerateNewWeightNode(func_graph, down_scale_cnode->fullname_with_scope() + "_weight");
  if (new_weight_param == nullptr) {
    MS_LOG(ERROR) << "Generate new weight parameter node failed.";
    return nullptr;
  }
  auto down_scale_prim = ops::GetOperator<ops::ScaleFusion>(down_scale_cnode->input(FIRST_INPUT));
  MS_CHECK_TRUE_RET(down_scale_prim != nullptr, nullptr);
  auto down_scale_prim_c = down_scale_prim->GetPrim();
  MS_CHECK_TRUE_RET(down_scale_prim_c != nullptr && down_scale_prim_c->GetAttr(ops::kAxis) != nullptr, nullptr);
  down_scale_prim->set_axis(MSMIN(up_scale_axis_, down_scale_axis_));

  auto manager = func_graph->manager();
  MS_ASSERT(manager != nullptr);
  manager->SetEdge(down_scale_cnode, 1, up_scale_cnode->input(SECOND_INPUT));
  manager->SetEdge(down_scale_cnode, kInputIndexTwo, new_weight_param);
  if (up_scale_cnode->size() == kScaleWithBiasLen) {
    ParameterPtr new_bias_param = GenerateNewBiasNode(func_graph, down_scale_cnode->fullname_with_scope() + "_bias");
    if (new_bias_param == nullptr) {
      MS_LOG(ERROR) << "Generate new weight parameter node failed.";
      return nullptr;
    }
    if (down_scale_cnode->size() == kScaleWithBiasLen) {
      manager->SetEdge(down_scale_cnode, kInputIndexThree, new_bias_param);
    } else {
      manager->AddEdge(down_scale_cnode, new_bias_param);
    }
  }

  return nullptr;
}
}  // namespace mindspore::opt
