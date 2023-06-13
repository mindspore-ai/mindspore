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

#include "bolt/bolt_parameter_manager.h"
#include "bolt/bolt_utils.h"
#include "nnacl/conv_parameter.h"
#include "schema/ops_generated.h"

namespace mindspore::kernel::bolt {
using mindspore::schema::PrimitiveType_Conv2DFusion;
ParameterSpec *PopulateConv2DBoltParameter(const OpParameter *op_parameter) {
  auto conv_param = reinterpret_cast<const ConvParameter *>(op_parameter);
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "Get nullptr OpParameter";
    return nullptr;
  }
  auto bolt_param = reinterpret_cast<ParameterSpec *>(malloc(sizeof(ParameterSpec)));
  if (bolt_param == nullptr) {
    MS_LOG(ERROR) << "Malloc ParameterSpec ptr failed";
    return nullptr;
  }
  ConvolutionParamSpec conv_param_spec;
  conv_param_spec.kernel_t = 1;
  conv_param_spec.kernel_h = conv_param->kernel_h_;
  conv_param_spec.kernel_w = conv_param->kernel_w_;
  conv_param_spec.stride_t = 1;
  conv_param_spec.stride_h = conv_param->stride_h_;
  conv_param_spec.stride_w = conv_param->stride_w_;
  conv_param_spec.dilatedRate_t = 1;
  conv_param_spec.dilatedRate_h = conv_param->dilation_h_;
  conv_param_spec.dilatedRate_w = conv_param->dilation_w_;
  conv_param_spec.pad_before = 0;
  conv_param_spec.pad_after = 0;
  conv_param_spec.pad_top = conv_param->pad_u_;
  conv_param_spec.pad_bottom = conv_param->pad_d_;
  conv_param_spec.pad_left = conv_param->pad_l_;
  conv_param_spec.pad_right = conv_param->pad_r_;
  conv_param_spec.num_outputs_origin = conv_param->output_channel_;
  conv_param_spec.num_outputs = UP_ROUND(conv_param->output_channel_, C8NUM);
  conv_param_spec.output_pad_t = 0;
  conv_param_spec.output_pad_h = 0;
  conv_param_spec.output_pad_w = 0;
  conv_param_spec.group = conv_param->group_;
  if (conv_param->group_ == 1) {
    conv_param_spec.convolution_type = CONVOLUTION_POINTWISE;
    auto ret = ConvertActType(conv_param->act_type_, &conv_param_spec.pw_activation_type);
    if (ret != lite::RET_OK) {
      return nullptr;
    }
  } else if (conv_param->group_ == conv_param->input_channel_ && conv_param->group_ == conv_param->output_channel_) {
    conv_param_spec.convolution_type = CONVOLUTION_DEPTHWISE;
    auto ret = ConvertActType(conv_param->act_type_, &conv_param_spec.dw_activation_type);
    if (ret != lite::RET_OK) {
      return nullptr;
    }
  } else {
    MS_LOG(ERROR) << "Unsupported convolution type for bolt convolution kernel.";
    return nullptr;
  }
  bolt_param->conv_spec = conv_param_spec;
  return bolt_param;
}
REG_BOLT_PARAMETER_POPULATE(PrimitiveType_Conv2DFusion, PopulateConv2DBoltParameter)
}  // namespace mindspore::kernel::bolt
