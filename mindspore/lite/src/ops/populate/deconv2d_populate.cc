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
#include "src/common/log_adapter.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/conv_parameter.h"
using mindspore::schema::PrimitiveType_Conv2dTransposeFusion;

namespace mindspore {
namespace lite {
OpParameter *PopulateDeconvParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_ASSERT(primitive != nullptr);
  auto value = primitive->value_as_Conv2dTransposeFusion();
  if (value == nullptr) {
    MS_LOG(ERROR) << "value is nullptr";
    return nullptr;
  }

  auto *param = reinterpret_cast<ConvParameter *>(malloc(sizeof(ConvParameter)));
  if (param == nullptr) {
    MS_LOG(ERROR) << "malloc ConvParameter failed.";
    return nullptr;
  }
  memset(param, 0, sizeof(ConvParameter));

  param->op_parameter_.type_ = primitive->value_type();
  auto kernel_size = value->kernel_size();
  auto stride = value->stride();
  auto pad_list = value->pad_list();
  auto dilation = value->dilation();
  auto output_paddings = value->output_paddings();
  if (kernel_size == nullptr || stride == nullptr || dilation == nullptr || output_paddings == nullptr) {
    MS_LOG(ERROR) << "nullptr";
    free(param);
    return nullptr;
  }
  param->kernel_h_ = static_cast<int>(*(kernel_size->begin()));
  param->kernel_w_ = static_cast<int>(*(kernel_size->begin() + 1));
  param->group_ = static_cast<int>(value->group());
  param->stride_h_ = static_cast<int>(*(stride->begin()));
  param->stride_w_ = static_cast<int>(*(stride->begin() + 1));
  param->output_padding_h_ = static_cast<int>(*(output_paddings->begin()));
  param->output_padding_w_ = static_cast<int>(*(output_paddings->begin() + 1));
  switch (value->pad_mode()) {
    case schema::PadMode_SAME:
      param->pad_mode_ = Pad_same;
      break;
    case schema::PadMode_VALID:
      param->pad_mode_ = Pad_valid;
      break;
    default:
      param->pad_mode_ = Pad_pad;
  }
  if (pad_list == nullptr || pad_list->size() < 4) {
    param->pad_u_ = 0;
    param->pad_d_ = 0;
    param->pad_l_ = 0;
    param->pad_r_ = 0;
  } else {
    param->pad_u_ = static_cast<int>(*(pad_list->begin()));
    param->pad_d_ = static_cast<int>(*(pad_list->begin() + 1));
    param->pad_l_ = static_cast<int>(*(pad_list->begin() + 2));
    param->pad_r_ = static_cast<int>(*(pad_list->begin() + 3));
  }
  param->dilation_h_ = static_cast<int>(*(dilation->begin()));
  param->dilation_w_ = static_cast<int>(*(dilation->begin() + 1));
  param->input_channel_ = static_cast<int>(value->in_channel());
  param->output_channel_ = static_cast<int>(value->out_channel());
  auto act_type = value->activation_type();
  switch (act_type) {
    case schema::ActivationType_RELU:
      param->act_type_ = ActType_Relu;
      break;
    case schema::ActivationType_RELU6:
      param->act_type_ = ActType_Relu6;
      break;
    default:
      param->act_type_ = ActType_No;
      break;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_Conv2dTransposeFusion, PopulateDeconvParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
