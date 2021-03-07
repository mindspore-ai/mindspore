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

#include "schema/model_v0_generated.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/conv_parameter.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateConvDwParameter(const void *prim) {
  auto *primitive = static_cast<const schema::v0::Primitive *>(prim);
  auto depthwise_conv2d_prim = primitive->value_as_DepthwiseConv2D();
  ConvParameter *conv_param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "malloc ConvParameter failed.";
    return nullptr;
  }
  memset(conv_param, 0, sizeof(ConvParameter));
  conv_param->op_parameter_.type_ = schema::PrimitiveType_Conv2DFusion;

  conv_param->group_ = depthwise_conv2d_prim->channelIn();

  conv_param->kernel_h_ = depthwise_conv2d_prim->kernelH();
  conv_param->kernel_w_ = depthwise_conv2d_prim->kernelW();
  conv_param->stride_h_ = depthwise_conv2d_prim->strideH();
  conv_param->stride_w_ = depthwise_conv2d_prim->strideW();

  conv_param->pad_u_ = depthwise_conv2d_prim->padUp();
  conv_param->pad_d_ = depthwise_conv2d_prim->padDown();
  conv_param->pad_l_ = depthwise_conv2d_prim->padLeft();
  conv_param->pad_r_ = depthwise_conv2d_prim->padRight();
  conv_param->input_channel_ = depthwise_conv2d_prim->channelIn();
  conv_param->output_channel_ = depthwise_conv2d_prim->channelIn();
  conv_param->dilation_h_ = depthwise_conv2d_prim->dilateH();
  conv_param->dilation_w_ = depthwise_conv2d_prim->dilateW();

  auto pad_mode = depthwise_conv2d_prim->padMode();
  switch (pad_mode) {
    case schema::v0::PadMode_SAME_UPPER:
      conv_param->pad_mode_ = Pad_same;
      break;
    case schema::v0::PadMode_VALID:
      conv_param->pad_mode_ = Pad_valid;
      break;
    default:
      conv_param->pad_mode_ = Pad_pad;
      break;
  }
  auto act_type = depthwise_conv2d_prim->activationType();
  switch (act_type) {
    case schema::v0::ActivationType_RELU:
      conv_param->act_type_ = ActType_Relu;
      break;
    case schema::v0::ActivationType_RELU6:
      conv_param->act_type_ = ActType_Relu6;
      break;
    default:
      conv_param->act_type_ = ActType_No;
      break;
  }
  conv_param->channel_multiplie_ = depthwise_conv2d_prim->channelMultiplier();
  return reinterpret_cast<OpParameter *>(conv_param);
}
}  // namespace

Registry g_depthwiseConv2DV0ParameterRegistry(schema::v0::PrimitiveType_DepthwiseConv2D, PopulateConvDwParameter,
                                              SCHEMA_V0);
}  // namespace lite
}  // namespace mindspore
