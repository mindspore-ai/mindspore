/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "tools/optimizer/fusion/conv_transform_fusion.h"
#include <algorithm>
#include <memory>
#include <vector>
#include "ops/fusion/conv2d_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "tools/common/tensor_util.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "tools/converter/quantizer/quant_param_holder.h"
#include "securec/include/securec.h"
#include "nnacl/op_base.h"
#include "ops/op_utils.h"

namespace mindspore::opt {
namespace {
constexpr size_t kConvWeightIndex = 2;
constexpr size_t kConvBiasIndex = 3;
constexpr size_t kConvNoBiasLen = 3;
constexpr size_t kConvWithBiasLen = 4;
int64_t GetOutChannels(const CNodePtr &conv_node) {
  MS_ASSERT(conv_node != nullptr);
  auto value_node = conv_node->input(0);
  MS_ASSERT(value_node != nullptr);
  if (CheckPrimitiveType(conv_node, prim::kPrimConv2DFusion)) {
    auto conv_prim = ops::GetOperator<ops::Conv2DFusion>(value_node);
    MS_ASSERT(conv_prim != nullptr);
    auto conv_prim_c = conv_prim->GetPrim();
    MS_ASSERT(conv_prim_c != nullptr);
    if (conv_prim_c->GetAttr(ops::kOutChannel) == nullptr) {
      return 0;
    }
    return conv_prim->get_out_channel();
  } else if (CheckPrimitiveType(conv_node, prim::kPrimConv2dTransposeFusion)) {
    auto conv_prim = ops::GetOperator<ops::Conv2dTransposeFusion>(value_node);
    MS_ASSERT(conv_prim != nullptr);
    auto conv_prim_c = conv_prim->GetPrim();
    MS_ASSERT(conv_prim_c != nullptr);
    if (conv_prim_c->GetAttr(ops::kOutChannel) == nullptr) {
      return 0;
    }
    return conv_prim->get_out_channel();
  }
  return 0;
}

void GenerateNewWeightConv2D(float *dst_weight, const float *conv_weight, const float *scale_weight,
                             size_t weight_shape_size, int kernel_num) {
  MS_ASSERT(dst_weight != nullptr && conv_weight != nullptr && scale_weight != nullptr);
  if (kernel_num <= 0) {
    return;
  }
  auto kernel_size = weight_shape_size / static_cast<size_t>(kernel_num);
  for (size_t i = 0; i < static_cast<size_t>(kernel_num); ++i) {
    for (size_t j = 0; j < kernel_size; j++) {
      dst_weight[i * kernel_size + j] = conv_weight[i * kernel_size + j] * scale_weight[i];
    }
  }
}

void GenerateNewWeightConv2DTranspose(float *dst_weight, const float *scale_weight,
                                      const tensor::TensorPtr &weight_tensor, int64_t group, int kernel_num) {
  MS_ASSERT(dst_weight != nullptr && scale_weight != nullptr && weight_tensor != nullptr);
  if (group <= 0 || kernel_num <= 0) {
    return;
  }
  MS_ASSERT(weight_tensor->data_c() != nullptr);
  auto weight_data = reinterpret_cast<float *>(weight_tensor->data_c());
  auto cin_group = weight_tensor->shape()[0] / group;
  int64_t area_size = weight_tensor->shape()[kNHWC_H] * weight_tensor->shape()[kNHWC_W];
  for (int64_t k = 0; k < cin_group; ++k) {
    for (int64_t j = 0; j < area_size; j++) {
      for (int64_t i = 0; i < kernel_num; ++i) {
        dst_weight[i + j * kernel_num + k * area_size * kernel_num] =
          weight_data[i + j * kernel_num + k * area_size * kernel_num] * scale_weight[i];
      }
    }
  }
}

// this function should replace GenerateNewWeightConv2DTranspose after all fusions support NCHW
void GenerateNewWeightConv2DTranspose_NCHW(float *dst_weight, const float *scale_weight,
                                           const tensor::TensorPtr &weight_tensor, int64_t group, int kernel_num) {
  MS_ASSERT(dst_weight != nullptr && scale_weight != nullptr && weight_tensor != nullptr);
  if (group <= 0 || kernel_num <= 0) {
    return;
  }
  auto cin_group = weight_tensor->shape()[0] / group;
  MS_ASSERT(weight_tensor->data_c() != nullptr);
  auto weight_data = reinterpret_cast<float *>(weight_tensor->data_c());
  int64_t area_size = weight_tensor->shape()[kNHWC_H] * weight_tensor->shape()[kNHWC_W];
  for (int64_t k = 0; k < cin_group; ++k) {
    for (int64_t i = 0; i < kernel_num; ++i) {   // output channel num -> C
      for (int64_t j = 0; j < area_size; j++) {  // HW
        dst_weight[i * area_size + j + k * area_size * kernel_num] =
          weight_data[i * area_size + j + k * area_size * kernel_num] * scale_weight[i];
      }
    }
  }
}
}  // namespace

const AnfNodePtr ConvTransformFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr) {
    return nullptr;
  }
  // transform node means scale,bn
  auto transform_node = node->cast<CNodePtr>();
  if (transform_node == nullptr || transform_node->size() < kInputSizeTwo) {
    return nullptr;
  }
  if (IsMarkedTrainOp(transform_node)) {
    return nullptr;
  }

  auto pre_node = transform_node->input(1);
  auto conv_node = pre_node->cast<CNodePtr>();
  MS_CHECK_TRUE_RET(conv_node != nullptr, nullptr);
  if (!CheckCanFused(func_graph, conv_node)) {
    return nullptr;
  }

  // Check the activation type of scale.
  if (!AdjustActivationType(conv_node, transform_node)) {
    return nullptr;
  }
  auto abstr = transform_node->abstract();
  int kernel_nums = static_cast<int>(GetOutChannels(conv_node));
  if (kernel_nums <= 0) {
    MS_LOG(INFO) << "Unsupported conv node, " << conv_node->DebugString();
    return node;
  }
  auto trans_scale = new (std::nothrow) float[kernel_nums];
  if (trans_scale == nullptr) {
    MS_LOG(ERROR) << "tensor_data is nullptr";
    return nullptr;
  }
  auto trans_bias = new (std::nothrow) float[kernel_nums];
  if (trans_bias == nullptr) {
    MS_LOG(ERROR) << "tensor_data is nullptr";
    delete[] trans_scale;
    return nullptr;
  }
  if (GenTransParam(transform_node, kernel_nums, trans_scale, trans_bias) != lite::RET_OK) {
    MS_LOG(DEBUG) << "cannot do fusion.";
    delete[] trans_bias;
    delete[] trans_scale;
    return nullptr;
  }
  if (GenNewConvTensor(func_graph, conv_node, kernel_nums, trans_scale, trans_bias) != lite::RET_OK) {
    MS_LOG(WARNING) << "generate a new weight tensor failed.";
    delete[] trans_bias;
    delete[] trans_scale;
    return nullptr;
  }
  delete[] trans_bias;
  delete[] trans_scale;
  pre_node->set_abstract(abstr);
  return pre_node;
}

bool ConvTransformFusion::AdjustActivationType(const CNodePtr &conv_node, const CNodePtr &transform_node) const {
  MS_ASSERT(conv_node != nullptr && transform_node != nullptr);
  MS_CHECK_TRUE_RET(transform_node->input(0) != nullptr, false);
  auto trans_prim = GetValueNode<PrimitivePtr>(transform_node->input(0));
  MS_CHECK_TRUE_RET(trans_prim != nullptr, false);
  auto trans_act_ptr = trans_prim->GetAttr(ops::kActivationType);
  if (trans_act_ptr == nullptr || GetValue<int64_t>(trans_act_ptr) == ActivationType::NO_ACTIVATION) {
    return true;
  }
  auto trans_act = GetValue<int64_t>(trans_act_ptr);
  // convolution only supports RELU and RELU6.
  if (trans_act != ActivationType::RELU && trans_act != ActivationType::RELU6) {
    return false;
  }
  MS_CHECK_TRUE_RET(conv_node->input(0) != nullptr, false);
  auto conv_prim = GetValueNode<PrimitivePtr>(conv_node->input(0));
  MS_CHECK_TRUE_RET(conv_prim != nullptr, false);
  (void)conv_prim->AddAttr(ops::kActivationType, MakeValue(trans_act));
  return true;
}

int ConvTransformFusion::GenTransParam(const CNodePtr &transform_node, int kernel_nums, float *trans_scale,
                                       float *trans_bias) const {
  MS_ASSERT(transform_node != nullptr);
  if (trans_scale == nullptr) {
    MS_LOG(ERROR) << "new transScale failed";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return lite::RET_NULL_PTR;
  }
  if (trans_bias == nullptr) {
    MS_LOG(ERROR) << "new transBias failed";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return lite::RET_NULL_PTR;
  }
  if (memset_s(trans_scale, kernel_nums * sizeof(float), 0, kernel_nums * sizeof(float)) != EOK) {
    MS_LOG(ERROR) << "memset transScale failed";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    return lite::RET_ERROR;
  }
  if (memset_s(trans_bias, kernel_nums * sizeof(float), 0, kernel_nums * sizeof(float)) != EOK) {
    MS_LOG(ERROR) << "memset transBias failed";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    return lite::RET_ERROR;
  }

  return InitTransParam(transform_node, kernel_nums, trans_scale, trans_bias);
}

int ConvTransformFusion::GenNewConvTensor(const FuncGraphPtr &func_graph, const CNodePtr &conv_node, int kernel_num,
                                          const float *trans_scale, const float *trans_bias) const {
  MS_ASSERT(func_graph != nullptr && conv_node != nullptr);
  MS_ASSERT(trans_scale != nullptr && trans_bias != nullptr);
  auto manager = func_graph->manager();
  MS_CHECK_TRUE_MSG(manager != nullptr, lite::RET_NULL_PTR, "manager is nullptr");
  AnfNodePtr conv_weight_node = nullptr;
  AnfNodePtr conv_bias_node = nullptr;
  if (conv_node->inputs().size() == kConvNoBiasLen) {
    conv_weight_node = conv_node->input(kConvWeightIndex);
  } else if (conv_node->inputs().size() == kConvWithBiasLen) {
    conv_weight_node = conv_node->input(kConvWeightIndex);
    conv_bias_node = conv_node->input(kConvBiasIndex);
  } else {
    MS_LOG(ERROR) << "conv node:" << conv_node->DebugString() << "inputs size must 3 or 4";
    return lite::RET_ERROR;
  }
  MS_CHECK_TRUE_RET(conv_weight_node != nullptr, lite::RET_ERROR);
  if (!conv_weight_node->isa<Parameter>()) {
    MS_LOG(ERROR) << "scale weight node not parameter node";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return lite::RET_ERROR;
  }
  if (conv_bias_node != nullptr && !conv_bias_node->isa<Parameter>()) {
    MS_LOG(ERROR) << "scale bias node not parameter node";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return lite::RET_ERROR;
  }
  auto conv_weight_param = conv_weight_node->cast<ParameterPtr>()->default_param();
  auto weight_tensor = std::dynamic_pointer_cast<tensor::Tensor>(conv_weight_param);
  if (kernel_num <= 0) {
    MS_LOG(ERROR) << "kernel num less than 0";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return lite::RET_ERROR;
  }
  auto new_weight_tensor = lite::CreateTensorInfo(weight_tensor->data_c(), weight_tensor->DataSize() * sizeof(float),
                                                  weight_tensor->shape(), weight_tensor->data_type());
  if (new_weight_tensor == nullptr) {
    MS_LOG(ERROR) << "create tensor info failed.";
    return lite::RET_ERROR;
  }
  if (CalNewWeightTensor(conv_node, new_weight_tensor, kernel_num, trans_scale) != lite::RET_OK) {
    MS_LOG(WARNING) << "generate a new weight tensor failed.";
    return lite::RET_ERROR;
  }
  float *bias_data = nullptr;
  // conv has bias,bias_flag true
  bool bias_flag = false;
  if (conv_bias_node != nullptr) {
    auto conv_bias_param = conv_bias_node->cast<ParameterPtr>()->default_param();
    auto bias_tensor = std::dynamic_pointer_cast<tensor::Tensor>(conv_bias_param);
    bias_data = reinterpret_cast<float *>(bias_tensor->data_c());
    bias_flag = true;
  } else {
    bias_data = new (std::nothrow) float[kernel_num];
    if (bias_data == nullptr) {
      MS_LOG(ERROR) << "tensor_data is nullptr";
      return lite::RET_ERROR;
    }
    if (memset_s(bias_data, kernel_num * sizeof(float), 0, kernel_num * sizeof(float)) != EOK) {
      delete[] bias_data;
      return lite::RET_ERROR;
    }
  }
  if (CalNewBiasTensor(bias_data, kernel_num, bias_flag, trans_scale, trans_bias) != lite::RET_OK) {
    MS_LOG(ERROR) << "generate a new bias failed.";
    if (!bias_flag) {
      delete[] bias_data;
    }
    return lite::RET_ERROR;
  }
  if (!bias_flag) {
    auto bias_node = AddNewBiasNode(bias_data, func_graph, kernel_num, weight_tensor->data_type());
    delete[] bias_data;
    bias_data = nullptr;
    if (bias_node == nullptr) {
      MS_LOG(ERROR) << "generate a new bias node failed.";
      return lite::RET_ERROR;
    }
    bias_node->set_name(conv_node->fullname_with_scope() + "_bias");
    manager->AddEdge(conv_node, bias_node);
  }
  auto new_weight_paramter = func_graph->add_parameter();
  if (new_weight_paramter == nullptr) {
    MS_LOG(ERROR) << "new_weight_paramter is nullptr";
    return lite::RET_ERROR;
  }
  new_weight_paramter->set_default_param(new_weight_tensor);
  new_weight_paramter->set_abstract(conv_weight_node->abstract());
  new_weight_paramter->set_name(conv_weight_node->fullname_with_scope());
  manager->SetEdge(conv_node, kConvWeightIndex, new_weight_paramter);
  return lite::RET_OK;
}

int ConvTransformFusion::CalNewWeightTensor(const CNodePtr &conv_node, const tensor::TensorPtr &weight_tensor,
                                            int kernel_num, const float *trans_scale) const {
  MS_ASSERT(conv_node != nullptr);
  MS_ASSERT(weight_tensor != nullptr);
  MS_ASSERT(trans_scale != nullptr);
  if (weight_tensor->shape().size() > kInputSizeFour) {
    MS_LOG(ERROR) << "weight tensor shape error";
    return lite::RET_ERROR;
  }
  auto weight_shape_size = weight_tensor->DataSize();
  MS_CHECK_TRUE_RET(weight_shape_size > 0, lite::RET_ERROR);
  auto tmp_weight_data = new (std::nothrow) float[weight_shape_size];
  if (tmp_weight_data == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    return lite::RET_ERROR;
  }
  auto data_size = weight_shape_size * sizeof(float);
  if (memset_s(tmp_weight_data, data_size, 0, data_size) != EOK) {
    MS_LOG(ERROR) << "memset newWeightData failed";
    delete[] tmp_weight_data;
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    return lite::RET_ERROR;
  }
  auto weight_data = reinterpret_cast<float *>(weight_tensor->data_c());
  auto conv_prim = GetValueNode<PrimitivePtr>(conv_node->input(0));
  MS_ASSERT(conv_prim != nullptr);
  bool is_depth_wise =
    conv_prim->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(conv_prim->GetAttr(ops::kIsDepthWise));
  if (CheckPrimitiveType(conv_node, prim::kPrimConv2DFusion)) {
    GenerateNewWeightConv2D(tmp_weight_data, weight_data, trans_scale, weight_shape_size, kernel_num);
  } else if (CheckPrimitiveType(conv_node, prim::kPrimConv2dTransposeFusion) && !is_depth_wise) {
    auto conv2d_prim = api::MakeShared<ops::Conv2dTransposeFusion>(conv_prim);
    MS_ASSERT(conv2d_prim != nullptr);
    auto conv2d_prim_c = conv2d_prim->GetPrim();
    MS_ASSERT(conv2d_prim_c != nullptr);
    auto group = conv2d_prim_c->GetAttr(ops::kGroup) == nullptr ? 1 : conv2d_prim->get_group();
    if (!nchw_format_) {
      GenerateNewWeightConv2DTranspose(tmp_weight_data, trans_scale, weight_tensor, group, kernel_num);
    } else {
      GenerateNewWeightConv2DTranspose_NCHW(tmp_weight_data, trans_scale, weight_tensor, group, kernel_num);
    }
  }
  auto ret = memcpy_s(weight_data, weight_tensor->Size(), tmp_weight_data, data_size);
  delete[] tmp_weight_data;
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy error: " << ret;
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    return lite::RET_ERROR;
  }
  return lite::RET_OK;
}

int ConvTransformFusion::CalNewBiasTensor(float *bias_data, int kernel_num, bool bias_flag, const float *trans_scale,
                                          const float *trans_bias) const {
  MS_ASSERT(bias_data != nullptr);
  MS_ASSERT(trans_bias != nullptr);
  MS_ASSERT(trans_scale != nullptr);
  if (bias_flag) {
    auto tmp_bias_data = new (std::nothrow) float[kernel_num];
    if (tmp_bias_data == nullptr) {
      MS_LOG(ERROR) << "tensor_data is nullptr";
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
      return lite::RET_NULL_PTR;
    }
    if (memset_s(tmp_bias_data, kernel_num * sizeof(float), 0, kernel_num * sizeof(float)) != EOK) {
      MS_LOG(ERROR) << "memset bias data failed";
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
      delete[] tmp_bias_data;
      return lite::RET_MEMORY_FAILED;
    }
    for (int i = 0; i < kernel_num; i++) {
      tmp_bias_data[i] = bias_data[i] * trans_scale[i] + trans_bias[i];
    }

    auto ret = memcpy_s(bias_data, kernel_num * sizeof(float), tmp_bias_data, kernel_num * sizeof(float));
    delete[] tmp_bias_data;
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy error: " << ret;
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
      return lite::RET_MEMORY_FAILED;
    }
  } else {
    if (memset_s(bias_data, kernel_num * sizeof(float), 0, kernel_num * sizeof(float)) != EOK) {
      MS_LOG(ERROR) << "memset bias data failed";
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
      return lite::RET_MEMORY_FAILED;
    }
    auto ret = memcpy_s(bias_data, kernel_num * sizeof(float), trans_bias, kernel_num * sizeof(float));
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy error: " << ret;
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
      return lite::RET_MEMORY_FAILED;
    }
  }
  return lite::RET_OK;
}

bool ConvTransformFusion::CheckCanFused(const FuncGraphPtr &func_graph, const CNodePtr &conv_node) const {
  MS_ASSERT(func_graph != nullptr && conv_node != nullptr);
  if (IsMultiOutputTensors(func_graph, conv_node) || IsMarkedTrainOp(conv_node)) {
    return false;
  }
  MS_ASSERT(conv_node->inputs().size() >= kConvNoBiasLen);
  auto conv_prim = GetValueNode<PrimitivePtr>(conv_node->input(kInputIndex));
  auto quant_attr = conv_prim->GetAttr("quant_params");
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
  auto conv_act_ptr = conv_prim->GetAttr(ops::kActivationType);
  if (conv_act_ptr != nullptr && GetValue<int64_t>(conv_act_ptr) != ActivationType::NO_ACTIVATION) {
    return false;
  }
  // Check weight is const.
  auto conv_weight_node = conv_node->input(kConvWeightIndex);
  bool is_value_node = conv_weight_node->isa<ValueNode>();
  auto conv_weight_param =
    conv_weight_node->isa<Parameter>() ? conv_weight_node->cast<ParameterPtr>()->default_param() : nullptr;
  return is_value_node || conv_weight_param != nullptr;
}
}  // namespace mindspore::opt
