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
#include "src/common/ops/populate/populate_register.h"
#include "nnacl/conv_parameter.h"
using mindspore::schema::PrimitiveType_Conv2dTransposeFusion;

namespace mindspore {
namespace lite {
int SetPadAndAct(schema::PadMode pad_mode, schema::ActivationType act_type, ConvParameter *param) {
  switch (pad_mode) {
    case schema::PadMode_SAME:
      param->pad_mode_ = Pad_same;
      break;
    case schema::PadMode_VALID:
      param->pad_mode_ = Pad_valid;
      break;
    case schema::PadMode_PAD:
      param->pad_mode_ = Pad_pad;
      break;
    default:
      MS_LOG(ERROR) << "pad mode does not support, " << pad_mode;
      return RET_NOT_SUPPORT;
  }

  switch (act_type) {
    case schema::ActivationType_RELU:
      param->act_type_ = ActType_Relu;
      break;
    case schema::ActivationType_RELU6:
      param->act_type_ = ActType_Relu6;
      break;
    default:
      if (act_type != schema::ActivationType_NO_ACTIVATION) {
        MS_LOG(ERROR) << "activation type does not support, " << act_type;
        return RET_NOT_SUPPORT;
      }
      param->act_type_ = ActType_No;
      break;
  }
  return RET_OK;
}

OpParameter *PopulateDeconvParameter(const void *prim) {
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  auto primitive = static_cast<const schema::Primitive *>(prim);
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
  param->kernel_h_ = -1;
  param->kernel_w_ = -1;
  if (kernel_size != nullptr) {
    if (kernel_size->size() < kMinShapeSizeTwo) {
      MS_LOG(ERROR) << "kernel size is invalid.";
      free(param);
      return nullptr;
    }
    CHECK_LESS_RETURN_RET(INT32_MAX, *(kernel_size->begin()), nullptr, param);
    param->kernel_h_ = static_cast<int>(*(kernel_size->begin()));
    CHECK_LESS_RETURN_RET(INT32_MAX, *(kernel_size->begin() + 1), nullptr, param);
    param->kernel_w_ = static_cast<int>(*(kernel_size->begin() + 1));
  }
  param->output_padding_h_ = 0;
  param->output_padding_w_ = 0;
  if (output_paddings != nullptr) {
    if (output_paddings->size() < kMinShapeSizeTwo) {
      MS_LOG(ERROR) << "output_paddings size is invalid.";
      free(param);
      return nullptr;
    }
    CHECK_LESS_RETURN_RET(INT32_MAX, *(output_paddings->begin()), nullptr, param);
    param->output_padding_h_ = static_cast<int>(*(output_paddings->begin()));
    CHECK_LESS_RETURN_RET(INT32_MAX, *(output_paddings->begin() + 1), nullptr, param);
    param->output_padding_w_ = static_cast<int>(*(output_paddings->begin() + 1));
  }
  if (param->output_padding_h_ < 0 || param->output_padding_w_ < 0) {
    MS_LOG(ERROR) << "invalid output padding";
    free(param);
    return nullptr;
  }

  if (stride == nullptr || dilation == nullptr) {
    MS_LOG(ERROR) << "nullptr";
    free(param);
    return nullptr;
  }
  if (stride->size() < kMinShapeSizeTwo || dilation->size() < kMinShapeSizeTwo) {
    MS_LOG(ERROR) << "stride size: " << stride->size() << ", dilation size: " << dilation->size();
    free(param);
    return nullptr;
  }

  CHECK_LESS_RETURN_RET(INT32_MAX, value->group(), nullptr, param);
  param->group_ = static_cast<int>(value->group());
  CHECK_LESS_RETURN_RET(INT32_MAX, *(stride->begin()), nullptr, param);
  param->stride_h_ = static_cast<int>(*(stride->begin()));
  CHECK_LESS_RETURN_RET(INT32_MAX, *(stride->begin() + 1), nullptr, param);
  param->stride_w_ = static_cast<int>(*(stride->begin() + 1));

  if (pad_list == nullptr || pad_list->size() < kMinShapeSizeFour) {
    param->pad_u_ = 0;
    param->pad_d_ = 0;
    param->pad_l_ = 0;
    param->pad_r_ = 0;
  } else {
    CHECK_LESS_RETURN_RET(INT32_MAX, *(pad_list->begin()), nullptr, param);
    param->pad_u_ = static_cast<int>(*(pad_list->begin()));
    CHECK_LESS_RETURN_RET(INT32_MAX, *(pad_list->begin() + 1), nullptr, param);
    param->pad_d_ = static_cast<int>(*(pad_list->begin() + 1));
    CHECK_LESS_RETURN_RET(INT32_MAX, *(pad_list->begin() + kOffsetTwo), nullptr, param);
    param->pad_l_ = static_cast<int>(*(pad_list->begin() + kOffsetTwo));
    CHECK_LESS_RETURN_RET(INT32_MAX, *(pad_list->begin() + kOffsetThree), nullptr, param);
    param->pad_r_ = static_cast<int>(*(pad_list->begin() + kOffsetThree));
  }
  CHECK_LESS_RETURN_RET(INT32_MAX, *(dilation->begin()), nullptr, param);
  param->dilation_h_ = static_cast<int>(*(dilation->begin()));

  CHECK_LESS_RETURN_RET(INT32_MAX, *(dilation->begin() + 1), nullptr, param);
  param->dilation_w_ = static_cast<int>(*(dilation->begin() + 1));

  CHECK_LESS_RETURN_RET(INT32_MAX, value->in_channel(), nullptr, param);
  param->input_channel_ = static_cast<int>(value->in_channel());

  CHECK_LESS_RETURN_RET(INT32_MAX, value->out_channel(), nullptr, param);
  param->output_channel_ = static_cast<int>(value->out_channel());

  auto act_type = value->activation_type();
  auto pad_mode = value->pad_mode();
  if (SetPadAndAct(pad_mode, act_type, param) != RET_OK) {
    MS_LOG(ERROR) << "SetPadAndAct failed.";
    free(param);
    return nullptr;
  }

  return reinterpret_cast<OpParameter *>(param);
}
REG_POPULATE(PrimitiveType_Conv2dTransposeFusion, PopulateDeconvParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
