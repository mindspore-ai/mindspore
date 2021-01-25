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

#include "nnacl/conv_parameter.h"
#include "src/ops/populate/populate_register.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateConvParameter(const void *prim) {
  ConvParameter *conv_param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (conv_param == nullptr) {
    MS_LOG(ERROR) << "malloc ConvParameter failed.";
    return nullptr;
  }
  memset(conv_param, 0, sizeof(ConvParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  conv_param->op_parameter_.type_ = primitive->value_type();
  auto conv_primitive = primitive->value_as_Conv2DFusion();
  conv_param->kernel_h_ = static_cast<int>(*(conv_primitive->kernel_size()->begin()));
  conv_param->kernel_w_ = static_cast<int>(*(conv_primitive->kernel_size()->begin() + 1));
  conv_param->group_ = static_cast<int>(conv_primitive->group());
  conv_param->stride_h_ = static_cast<int>(*(conv_primitive->stride()->begin()));
  conv_param->stride_w_ = static_cast<int>(*(conv_primitive->stride()->begin() + 1));
  switch (conv_primitive->pad_mode()) {
    case schema::PadMode_SAME:
      conv_param->pad_mode_ = Pad_same;
      break;
    case schema::PadMode_VALID:
      conv_param->pad_mode_ = Pad_valid;
      break;
    default:
      conv_param->pad_mode_ = Pad_pad;
  }
  if (conv_primitive->pad_list() == nullptr || conv_primitive->pad_list()->size() < 4) {
    conv_param->pad_u_ = 0;
    conv_param->pad_d_ = 0;
    conv_param->pad_l_ = 0;
    conv_param->pad_r_ = 0;
  } else {
    conv_param->pad_u_ = static_cast<int>(*(conv_primitive->pad_list()->begin()));
    conv_param->pad_d_ = static_cast<int>(*(conv_primitive->pad_list()->begin() + 1));
    conv_param->pad_l_ = static_cast<int>(*(conv_primitive->pad_list()->begin() + 2));
    conv_param->pad_r_ = static_cast<int>(*(conv_primitive->pad_list()->begin() + 3));
  }
  conv_param->dilation_h_ = static_cast<int>(*(conv_primitive->dilation()->begin()));
  conv_param->dilation_w_ = static_cast<int>(*(conv_primitive->dilation()->begin() + 1));
  conv_param->input_channel_ = static_cast<int>(conv_primitive->in_channel());
  conv_param->output_channel_ = static_cast<int>(conv_primitive->out_channel());
  auto act_type = conv_primitive->activation_type();
  switch (act_type) {
    case schema::ActivationType_RELU:
      conv_param->act_type_ = ActType_Relu;
      break;
    case schema::ActivationType_RELU6:
      conv_param->act_type_ = ActType_Relu6;
      break;
    default:
      conv_param->act_type_ = ActType_No;
  }
  return reinterpret_cast<OpParameter *>(conv_param);
}
}  // namespace
Registry g_conv2DParameterRegistry(schema::PrimitiveType_Conv2DFusion, PopulateConvParameter, SCHEMA_CUR);
}  // namespace lite
}  // namespace mindspore
