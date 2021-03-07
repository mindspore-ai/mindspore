/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "src/ops/populate/populate_register.h"
#include "nnacl/conv_parameter.h"

namespace mindspore {
namespace lite {

OpParameter *PopulateConvDwParameter(const void *primitive) {
  ConvParameter *conv_param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "malloc ConvParameter failed.";
    return nullptr;
  }
  memset(conv_param, 0, sizeof(ConvParameter));
  // conv_param->op_parameter_.type_ = primitive->Type();

  // auto conv_primitive =
  //   reinterpret_cast<mindspore::lite::DepthwiseConv2D *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  // conv_param->kernel_h_ = conv_primitive->GetKernelH();
  // conv_param->kernel_w_ = conv_primitive->GetKernelW();
  // conv_param->stride_h_ = conv_primitive->GetStrideH();
  // conv_param->stride_w_ = conv_primitive->GetStrideW();

  // auto convdw_lite_primitive = (lite::DepthwiseConv2D *)primitive;
  // conv_param->pad_u_ = convdw_lite_primitive->PadUp();
  // conv_param->pad_d_ = convdw_lite_primitive->PadDown();
  // conv_param->pad_l_ = convdw_lite_primitive->PadLeft();
  // conv_param->pad_r_ = convdw_lite_primitive->PadRight();
  // conv_param->input_channel_ = convdw_lite_primitive->GetInputChannel();
  // conv_param->dilation_h_ = conv_primitive->GetDilateH();
  // conv_param->dilation_w_ = conv_primitive->GetDilateW();
  // auto pad_mode = conv_primitive->GetPadMode();
  // switch (pad_mode) {
  //   case schema::PadMode_SAME_UPPER:
  //     conv_param->pad_mode_ = Pad_Same;
  //     break;
  //   case schema::PadMode_VALID:
  //     conv_param->pad_mode_ = Pad_Valid;
  //     break;
  //   default:
  //     conv_param->pad_mode_ = Pad_No;
  //     break;
  // }
  // auto act_type = conv_primitive->GetActivationType();
  // switch (act_type) {
  //   case schema::ActivationType_RELU:
  //     conv_param->act_type_ = ActType_Relu;
  //     break;
  //   case schema::ActivationType_RELU6:
  //     conv_param->act_type_ = ActType_Relu6;
  //     break;
  //   default:
  //     conv_param->act_type_ = ActType_No;
  //     break;
  // }
  return reinterpret_cast<OpParameter *>(conv_param);
}
}  // namespace lite
}  // namespace mindspore
