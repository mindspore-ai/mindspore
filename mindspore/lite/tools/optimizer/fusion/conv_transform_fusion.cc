/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *conv_activation_fusion.h
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tools/optimizer/fusion/conv_transform_fusion.h"
#include <memory>
#include "ops/fusion/conv2d_fusion.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
#include "src/param_value_lite.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "securec/include/securec.h"

namespace mindspore::opt {
namespace {
constexpr size_t kConvWeightIndex = 2;
constexpr size_t kConvBiasIndex = 3;
constexpr size_t kConvNoBiasLen = 3;
constexpr size_t kConvWithBiasLen = 4;
int GetOutChannels(const CNodePtr &conv_node) {
  MS_ASSERT(conv_node != nullptr);
  auto value_node = conv_node->input(0);
  MS_ASSERT(value_node != nullptr);
  if (CheckPrimitiveType(conv_node, prim::kPrimConv2DFusion)) {
    auto conv_prim = GetValueNode<std::shared_ptr<ops::Conv2DFusion>>(value_node);
    MS_ASSERT(conv_prim != nullptr);
    if (conv_prim->GetAttr(ops::kOutChannel) == nullptr) {
      return 0;
    }
    return conv_prim->get_out_channel();
  } else if (CheckPrimitiveType(conv_node, prim::kPrimConv2dTransposeFusion)) {
    auto conv_prim = GetValueNode<std::shared_ptr<ops::Conv2dTransposeFusion>>(value_node);
    MS_ASSERT(conv_prim != nullptr);
    if (conv_prim->GetAttr(ops::kOutChannel) == nullptr) {
      return 0;
    }
    return conv_prim->get_out_channel();
  }
  return 0;
}

void GenerateNewWeightConv2D(float *dst_weight, const float *conv_weight, const float *scale_weight, FmkType fmk,
                             int weight_shape_size, int kernel_num) {
  if (dst_weight == nullptr || conv_weight == nullptr || scale_weight == nullptr) {
    return;
  }
  if (fmk == lite::converter::FmkType_TF) {
    for (int i = 0; i < weight_shape_size; i++) {
      dst_weight[i] = conv_weight[i] * scale_weight[i % kernel_num];
    }
  } else {
    MS_ASSERT(kernel_num > 0);
    auto kernel_size = weight_shape_size / kernel_num;
    for (int i = 0; i < kernel_num; i++) {
      for (int j = 0; j < kernel_size; j++) {
        dst_weight[i * kernel_size + j] = conv_weight[i * kernel_size + j] * scale_weight[i];
      }
    }
  }
}

void GenerateNewWeightConv2DTranspose(float *dst_weight, const float *scale_weight,
                                      const ParamValueLitePtr &weight_tensor, FmkType fmk, int group, int kernel_num) {
  if (dst_weight == nullptr || scale_weight == nullptr || weight_tensor == nullptr) {
    return;
  }
  auto weight_data = reinterpret_cast<float *>(weight_tensor->tensor_addr());
  if (fmk == lite::converter::FmkType_TF) {
    auto cin_group = weight_tensor->tensor_shape()[3] / group;
    int area_size = weight_tensor->tensor_shape()[0] * weight_tensor->tensor_shape()[1];
    for (int j = 0; j < area_size; j++) {
      for (int i = 0; i < kernel_num; ++i) {
        for (int k = 0; k < cin_group; ++k) {
          dst_weight[k + i * cin_group + j * kernel_num * cin_group] =
            weight_data[k + i * cin_group + j * kernel_num * cin_group] * scale_weight[i];
        }
      }
    }
  } else {
    MS_ASSERT(group > 0);
    auto cin_group = weight_tensor->tensor_shape()[0] / group;
    int area_size = weight_tensor->tensor_shape()[2] * weight_tensor->tensor_shape()[3];
    int cout_size = kernel_num * area_size;
    for (int k = 0; k < cin_group; ++k) {
      for (int i = 0; i < kernel_num; ++i) {
        auto row_addr = weight_data + k * cout_size + i * area_size;
        auto new_row_addr = dst_weight + k * cout_size + i * area_size;
        for (int j = 0; j < area_size; j++) {
          new_row_addr[j] = row_addr[j] * scale_weight[i];
        }
      }
    }
  }
}
}  // namespace

const AnfNodePtr ConvTransformFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  MS_LOG(DEBUG) << "conv activation pass process";
  if (CheckIfFuncGraphIsNull(func_graph) != lite::RET_OK || CheckIfAnfNodeIsNull(node) != lite::RET_OK) {
    return nullptr;
  }
  // transform node means scale,bn
  auto transform_node = node->cast<CNodePtr>();
  if (CheckIfCNodeIsNull(transform_node) != lite::RET_OK || CheckLeastInputSize(transform_node, 2) != lite::RET_OK) {
    return nullptr;
  }

  auto pre_node = transform_node->input(1);
  auto conv_node = pre_node->cast<CNodePtr>();
  if (IsMultiOutputTensors(func_graph, conv_node)) {
    return nullptr;
  }

  auto abstr = transform_node->abstract();
  int kernel_nums = GetOutChannels(conv_node);
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
  GenTransParam(transform_node, kernel_nums, trans_scale, trans_bias);
  GenNewConvTensor(func_graph, conv_node, kernel_nums, trans_scale, trans_bias);
  delete[] trans_bias;
  delete[] trans_scale;
  pre_node->set_abstract(abstr);
  return pre_node;
}

void ConvTransformFusion::GenTransParam(const CNodePtr &transform_node, int kernel_nums, float *trans_scale,
                                        float *trans_bias) const {
  if (trans_scale == nullptr) {
    MS_LOG(ERROR) << "new transScale failed";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return;
  }
  if (trans_bias == nullptr) {
    MS_LOG(ERROR) << "new transBias failed";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_NULL_PTR);
    return;
  }
  if (0 != memset_s(trans_scale, kernel_nums * sizeof(float), 0, kernel_nums * sizeof(float))) {
    MS_LOG(ERROR) << "memset transScale failed";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    return;
  }
  if (0 != memset_s(trans_bias, kernel_nums * sizeof(float), 0, kernel_nums * sizeof(float))) {
    MS_LOG(ERROR) << "memset transBias failed";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    return;
  }

  InitTransParam(transform_node, kernel_nums, trans_scale, trans_bias);
}

void ConvTransformFusion::GenNewConvTensor(const FuncGraphPtr &func_graph, const CNodePtr &conv_node, int kernel_num,
                                           const float *trans_scale, const float *trans_bias) const {
  MS_ASSERT(conv_node != nullptr);
  AnfNodePtr conv_weight_node = nullptr;
  AnfNodePtr conv_bias_node = nullptr;
  if (conv_node->inputs().size() == kConvNoBiasLen) {
    conv_weight_node = conv_node->input(kConvWeightIndex);
  } else if (conv_node->inputs().size() == kConvWithBiasLen) {
    conv_weight_node = conv_node->input(kConvWeightIndex);
    conv_bias_node = conv_node->input(kConvBiasIndex);
  } else {
    MS_LOG(ERROR) << "conv node:" << conv_node->DebugString() << "inputs size must 3 or 4";
    return;
  }
  if (!conv_weight_node->isa<Parameter>()) {
    MS_LOG(ERROR) << "scale weight node not parameter node";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return;
  }
  if (conv_bias_node != nullptr && !conv_bias_node->isa<Parameter>()) {
    MS_LOG(ERROR) << "scale bias node not parameter node";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return;
  }
  auto conv_weight_param = conv_weight_node->cast<ParameterPtr>()->default_param();
  auto weight_tensor = std::dynamic_pointer_cast<ParamValueLite>(conv_weight_param);
  if (kernel_num <= 0) {
    MS_LOG(ERROR) << "kernel num less than 0";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_INVALID_OP_ATTR);
    return;
  }
  auto temp_weight_data = new (std::nothrow) float[weight_tensor->tensor_shape_size()];
  if (temp_weight_data == nullptr) {
    MS_LOG(ERROR) << "new ParamValueLite failed";
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_ERROR);
    return;
  }
  auto new_weight_tensor = std::make_shared<ParamValueLite>();
  if (new_weight_tensor == nullptr) {
    delete temp_weight_data;
    MS_LOG(ERROR) << "new ParamValueLite failed";
    return;
  }
  new_weight_tensor->set_tensor_size(weight_tensor->tensor_size());
  new_weight_tensor->set_tensor_shape(weight_tensor->tensor_shape());
  new_weight_tensor->set_tensor_type(weight_tensor->tensor_type());
  new_weight_tensor->set_format(weight_tensor->format());
  auto ret = memcpy_s(temp_weight_data, weight_tensor->tensor_shape_size() * sizeof(float),
                      weight_tensor->tensor_addr(), weight_tensor->tensor_shape_size() * sizeof(float));
  if (ret != EOK) {
    delete temp_weight_data;
    MS_LOG(ERROR) << "memcpy_s error:" << ret;
    return;
  }
  new_weight_tensor->SetTensorData(temp_weight_data, new_weight_tensor->tensor_size());
  CalNewWeightTensor(conv_node, new_weight_tensor, kernel_num, trans_scale);
  float *bias_data = nullptr;
  // conv has bias,bias_flag true
  bool bias_flag = false;
  if (conv_bias_node != nullptr) {
    auto conv_bias_param = conv_bias_node->cast<ParameterPtr>()->default_param();
    auto bias_tensor = std::dynamic_pointer_cast<ParamValueLite>(conv_bias_param);
    bias_data = reinterpret_cast<float *>(bias_tensor->tensor_addr());
    bias_flag = true;
  } else {
    bias_data = new (std::nothrow) float[kernel_num];
    if (bias_data == nullptr) {
      MS_LOG(ERROR) << "tensor_data is nullptr";
      delete temp_weight_data;
      return;
    }
  }
  CalNewBiasTensor(bias_data, kernel_num, bias_flag, trans_scale, trans_bias);
  if (!bias_flag) {
    auto bias_node = AddNewBiasNode(bias_data, func_graph, kernel_num, weight_tensor);
    bias_node->set_name(conv_node->fullname_with_scope() + "_bias");
    conv_node->add_input(bias_node);
  }
  auto new_weight_paramter = func_graph->add_parameter();
  if (new_weight_paramter == nullptr) {
    MS_LOG(ERROR) << "new_weight_paramter is nullptr";
    delete temp_weight_data;
    return;
  }
  new_weight_paramter->set_default_param(new_weight_tensor);
  new_weight_paramter->set_abstract(conv_weight_node->abstract());
  new_weight_paramter->set_name(conv_node->fullname_with_scope() + conv_weight_node->fullname_with_scope());
  conv_node->set_input(kConvWeightIndex, new_weight_paramter);
}

void ConvTransformFusion::CalNewWeightTensor(const CNodePtr &conv_node, const ParamValueLitePtr &weight_tensor,
                                             int kernel_num, const float *trans_scale) const {
  MS_ASSERT(weight_data != nullptr);
  MS_ASSERT(trans_scale != nullptr);
  if (weight_tensor->tensor_shape().size() != 4) {
    MS_LOG(ERROR) << "weight tensor shape error";
    return;
  }
  auto weight_shape_size = weight_tensor->tensor_shape_size();
  auto tmp_weight_data = new (std::nothrow) float[weight_shape_size];
  if (tmp_weight_data == nullptr) {
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    return;
  }
  MS_ASSERT(new_weight_data != nullptr);
  auto data_size = weight_shape_size * sizeof(float);
  if (0 != memset_s(tmp_weight_data, data_size, 0, data_size)) {
    MS_LOG(ERROR) << "memset newWeightData failed";
    delete[] tmp_weight_data;
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    return;
  }
  auto weight_data = reinterpret_cast<float *>(weight_tensor->tensor_addr());
  auto conv_prim = GetValueNode<PrimitivePtr>(conv_node->input(0));
  MS_ASSERT(conv_prim != nullptr);
  bool is_depth_wise =
    conv_prim->GetAttr(ops::kIsDepthWise) != nullptr && GetValue<bool>(conv_prim->GetAttr(ops::kIsDepthWise));
  if (CheckPrimitiveType(conv_node, prim::kPrimConv2DFusion)) {
    GenerateNewWeightConv2D(tmp_weight_data, weight_data, trans_scale, fmk_type_, weight_shape_size, kernel_num);
  } else if (CheckPrimitiveType(conv_node, prim::kPrimConv2dTransposeFusion) && !is_depth_wise) {
    auto conv_primc = conv_prim->cast<std::shared_ptr<ops::Conv2dTransposeFusion>>();
    MS_ASSERT(conv_primc != nullptr);
    auto group = conv_primc->GetAttr(ops::kGroup) == nullptr ? 1 : conv_primc->get_group();
    GenerateNewWeightConv2DTranspose(tmp_weight_data, trans_scale, weight_tensor, fmk_type_, group, kernel_num);
  }
  auto ret = memcpy_s(weight_data, data_size, tmp_weight_data, data_size);
  if (ret != EOK) {
    MS_LOG(ERROR) << "memcpy error: " << ret;
    lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    delete[] tmp_weight_data;
    return;
  }
  delete[] tmp_weight_data;
}

void ConvTransformFusion::CalNewBiasTensor(float *bias_data, int kernel_num, bool bias_flag, const float *trans_scale,
                                           const float *trans_bias) const {
  MS_ASSERT(bias_data != nullptr);
  MS_ASSERT(trans_bias != nullptr);
  MS_ASSERT(trans_scale != nullptr);
  if (bias_flag) {
    auto tmp_bias_data = new (std::nothrow) float[kernel_num];
    if (tmp_bias_data == nullptr) {
      MS_LOG(ERROR) << "tensor_data is nullptr";
      return;
    }
    if (EOK != memset_s(tmp_bias_data, kernel_num * sizeof(float), 0, kernel_num * sizeof(float))) {
      MS_LOG(ERROR) << "memset bias data failed";
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
      delete[] tmp_bias_data;
      return;
    }
    for (int i = 0; i < kernel_num; i++) {
      tmp_bias_data[i] = bias_data[i] * trans_scale[i] + trans_bias[i];
    }

    auto ret = memcpy_s(bias_data, kernel_num * sizeof(float), tmp_bias_data, kernel_num * sizeof(float));
    delete[] tmp_bias_data;
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy error: " << ret;
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
      return;
    }
  } else {
    if (EOK != memset_s(bias_data, kernel_num * sizeof(float), 0, kernel_num * sizeof(float))) {
      MS_LOG(ERROR) << "memset bias data failed";
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
      return;
    }
    auto ret = memcpy_s(bias_data, kernel_num * sizeof(float), trans_bias, kernel_num * sizeof(float));
    if (ret != EOK) {
      MS_LOG(ERROR) << "memcpy error: " << ret;
      lite::ReturnCode::GetSingleReturnCode()->UpdateReturnCode(lite::RET_MEMORY_FAILED);
    }
  }
}
}  // namespace mindspore::opt
