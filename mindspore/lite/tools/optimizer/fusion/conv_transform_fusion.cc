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
#include "src/param_value_lite.h"
#include "schema/inner/model_generated.h"
#include "src/ir/primitive_t_value.h"
#include "utils/utils.h"
#include "tools/optimizer/common/gllo_utils.h"
#include "include/errorcode.h"
#include "securec/include/securec.h"

namespace mindspore::opt {
namespace {
constexpr size_t kConvWeightIndex = 2;
constexpr size_t kConvBiasIndex = 3;
constexpr size_t kConvNoBiasLen = 3;
constexpr size_t kConvWithBiasLen = 4;

int Get_Kenrnel_nums(const CNodePtr &conv_node) {
  MS_ASSERT(conv_node != nullptr);
  auto value_primitive = conv_node->input(0);
  auto value_node = value_primitive->cast<ValueNodePtr>();
  MS_ASSERT(value_node != nullptr);
  auto value = value_node->value();
  MS_ASSERT(value != nullptr);
  auto primitive = value->cast<PrimitiveTValuePtr>();
  MS_ASSERT(primitive != nullptr);
  auto type = primitive->GetPrimitiveT()->value.type;
  if (type == schema::PrimitiveType_Conv2D) {
    return primitive->GetPrimitiveT()->value.AsConv2D()->channelOut;
  } else if (type == schema::PrimitiveType_DepthwiseConv2D) {
    return primitive->GetPrimitiveT()->value.AsDepthwiseConv2D()->channelMultiplier
        * primitive->GetPrimitiveT()->value.AsDepthwiseConv2D()->channelIn;
  } else {
    MS_LOG(ERROR) << "Unsupported opType, " << type;
    return 0;
  }
}
}  // namespace

const AnfNodePtr ConvTransformFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                              const EquivPtr &) const {
  MS_LOG(DEBUG) << "conv activation pass process";
  CheckIfFuncGraphIsNull(func_graph);

  CheckIfAnfNodeIsNull(node);
  // transform node means scale,bn
  auto transform_node = node->cast<CNodePtr>();
  CheckIfCNodeIsNull(transform_node);
  CheckLeastInputSize(transform_node, 2);

  auto pre_node = transform_node->input(1);
  auto conv_node = pre_node->cast<CNodePtr>();
  if (IsMultiOutputTensors(func_graph, conv_node)) {
    return transform_node;
  }
  int kernel_nums = Get_Kenrnel_nums(conv_node);
  if (kernel_nums <= 0) {
    MS_LOG(ERROR) << "Unsupported conv node, " << conv_node->DebugString();
    return node;
  }
  auto trans_scale = new(std::nothrow) float[kernel_nums];
  auto trans_bias = new(std::nothrow) float[kernel_nums];
  GenTransParam(transform_node, kernel_nums, trans_scale, trans_bias);
  GenNewConvTensor(func_graph, conv_node, kernel_nums, trans_scale, trans_bias);
  delete[] trans_bias;
  delete[] trans_scale;
  auto primitiveT_value = GetValueNode<std::shared_ptr<lite::PrimitiveTValue>>(conv_node->input(0));
  MS_ASSERT(primitiveT_value != nullptr);
  auto type = primitiveT_value->GetPrimitiveT()->value.type;
  if (type == schema::PrimitiveType_Conv2D) {
    primitiveT_value->GetPrimitiveT()->value.AsConv2D()->hasBias = true;
  } else if (type == schema::PrimitiveType_DepthwiseConv2D) {
    primitiveT_value->GetPrimitiveT()->value.AsDepthwiseConv2D()->hasBias = true;
  } else {
    MS_LOG(EXCEPTION) << "Unsupported opType, " << type;
  }
  return pre_node;
}

const void ConvTransformFusion::GenTransParam(const CNodePtr &transform_node, int kernel_nums,
                                              float *trans_scale, float *trans_bias) const {
  if (trans_scale == nullptr) {
    MS_LOG(EXCEPTION) << "new transScale failed";
  }
  if (trans_bias == nullptr) {
    MS_LOG(EXCEPTION) << "new transBias failed";
  }
  if (0 != memset_s(trans_scale, kernel_nums * sizeof(float), 0, kernel_nums * sizeof(float))) {
    MS_LOG(EXCEPTION) << "memset transScale failed";
  }
  if (0 != memset_s(trans_bias, kernel_nums * sizeof(float), 0, kernel_nums * sizeof(float))) {
    MS_LOG(EXCEPTION) << "memset transBias failed";
  }

  InitTransParam(transform_node, kernel_nums, trans_scale, trans_bias);
}

const void ConvTransformFusion::GenNewConvTensor(const FuncGraphPtr &func_graph, const CNodePtr &conv_node,
                                                 int kernel_num, const float *trans_scale, const float *trans_bias)
const {
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
    MS_LOG(EXCEPTION) << "scale weight node not paramter node";
  }
  if (conv_bias_node != nullptr && !conv_bias_node->isa<Parameter>()) {
    MS_LOG(EXCEPTION) << "scale bias node not paramter node";
  }

  auto conv_weight_param = conv_weight_node->cast<ParameterPtr>()->default_param();
  auto weight_tensor = std::dynamic_pointer_cast<ParamValueLite>(conv_weight_param);
  auto weight_data = reinterpret_cast<float *>(weight_tensor->tensor_addr());
  if (kernel_num <= 0) {
    MS_LOG(EXCEPTION) << "kernel num less than 0";
  }
  auto kernel_size = weight_tensor->tensor_shape_size() / kernel_num;

  CalNewWeightTensor(weight_data, kernel_num, kernel_size, trans_scale);

  float *bias_data = nullptr;
  // conv has bias,bias_flag true
  bool bias_flag = false;
  if (conv_bias_node != nullptr) {
    auto conv_bias_param = conv_bias_node->cast<ParameterPtr>()->default_param();
    auto bias_tensor = std::dynamic_pointer_cast<ParamValueLite>(conv_bias_param);
    bias_data = reinterpret_cast<float *>(bias_tensor->tensor_addr());
    bias_flag = true;
  } else {
    bias_data = new(std::nothrow) float[kernel_num];
  }
  CalNewBiasTensor(bias_data, kernel_num, bias_flag, trans_scale, trans_bias);
  if (!bias_flag) {
    auto bias_node = AddNewBiasNode(bias_data, func_graph, kernel_num, weight_tensor);
    conv_node->add_input(bias_node);
  }
}
const void ConvTransformFusion::CalNewWeightTensor(float *weight_data, int kernel_num, int kernel_size,
                                                   const float *trans_scale) const {
  MS_ASSERT(weight_data != nullptr);
  auto tmp_weight_data = new(std::nothrow) float[kernel_num * kernel_size];
  MS_ASSERT(new_weight_data != nullptr);
  auto data_size = kernel_num * kernel_size * sizeof(float);
  if (0 != memset_s(tmp_weight_data, data_size, 0, data_size)) {
    MS_LOG(EXCEPTION) << "memset newWeightData failed";
    delete[] tmp_weight_data;
    return;
  }

  for (size_t i = 0; i < kernel_num; i++) {
    for (size_t j = 0; j < kernel_size; j++) {
      tmp_weight_data[i * kernel_size + j] = weight_data[i * kernel_size + j] * trans_scale[i];
    }
  }

  auto ret = memcpy_s(weight_data, data_size, tmp_weight_data, data_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "memcpy error: " << ret;
  }

  delete[] tmp_weight_data;
}
const void ConvTransformFusion::CalNewBiasTensor(float *bias_data, int kernel_num, bool bias_flag,
                                                 const float *trans_scale, const float *trans_bias) const {
  MS_ASSERT(bias_data != nullptr);
  if (bias_flag) {
    auto tmp_bias_data = new(std::nothrow) float[kernel_num];
    if (EOK != memset_s(tmp_bias_data, kernel_num * sizeof(float), 0, kernel_num * sizeof(float))) {
      MS_LOG(EXCEPTION) << "memset bias data failed";
    }
    for (size_t i = 0; i < kernel_num; i++) {
      tmp_bias_data[i] = bias_data[i] * trans_scale[i] + trans_bias[i];
    }

    auto ret = memcpy_s(bias_data, kernel_num * sizeof(float), tmp_bias_data, kernel_num * sizeof(float));
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy error: " << ret;
    }
    delete[] tmp_bias_data;
  } else {
    if (EOK != memset_s(bias_data, kernel_num * sizeof(float), 0, kernel_num * sizeof(float))) {
      MS_LOG(EXCEPTION) << "memset bias data failed";
    }
    auto ret = memcpy_s(bias_data, kernel_num * sizeof(float), trans_bias, kernel_num * sizeof(float));
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "memcpy error: " << ret;
    }
  }
}
}  // namespace mindspore::opt
