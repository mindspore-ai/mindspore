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
#include "src/common/ops/populate/populate_register.h"
using mindspore::schema::PrimitiveType_Conv2DFusion;

namespace mindspore {
namespace lite {
namespace {
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
      MS_LOG(ERROR) << "Pad mode does not support, " << pad_mode;
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
}  // namespace

OpParameter *PopulateConvParameter(const void *prim) {
  auto primitive = static_cast<const schema::Primitive *>(prim);
  MS_CHECK_TRUE_MSG(primitive != nullptr, nullptr, "primitive is nullptr.");
  auto value = primitive->value_as_Conv2DFusion();
  MS_CHECK_TRUE_MSG(value != nullptr, nullptr, "value is nullptr.");

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
  if (kernel_size != nullptr) {
    if (kernel_size->size() < kMinShapeSizeTwo) {
      MS_LOG(ERROR) << "kernel size is invalid.";
      free(param);
      return nullptr;
    }
    param->kernel_h_ = static_cast<int>(*(kernel_size->begin()));
    param->kernel_w_ = static_cast<int>(*(kernel_size->begin() + 1));
  } else {
    param->kernel_h_ = -1;
    param->kernel_w_ = -1;
  }
  if (stride == nullptr || dilation == nullptr) {
    MS_LOG(ERROR) << "kernel_size/stride/dilation is nullptr";
    free(param);
    return nullptr;
  }
  if (stride->size() < kMinShapeSizeTwo || dilation->size() < kMinShapeSizeTwo) {
    MS_LOG(ERROR) << "stride size: " << stride->size() << ", dilation size: " << dilation->size();
    free(param);
    return nullptr;
  }
  for (size_t i = 0; i <= 1; i++) {
    auto stride_item = *(stride->begin() + i);
    if (stride_item < 0 || stride_item > static_cast<int64_t>(INT32_MAX)) {
      MS_LOG(ERROR) << "strides has invalid num.";
      free(param);
      return nullptr;
    }
  }
  param->group_ = static_cast<int>(value->group());
  param->stride_h_ = static_cast<int>(*(stride->begin()));
  param->stride_w_ = static_cast<int>(*(stride->begin() + 1));
  if (pad_list == nullptr || pad_list->size() < kMinShapeSizeFour) {
    param->pad_u_ = 0;
    param->pad_d_ = 0;
    param->pad_l_ = 0;
    param->pad_r_ = 0;
  } else {
    for (size_t i = 0; i <= kOffsetThree; i++) {
      auto pad_item = *(pad_list->begin() + i);
      if (pad_item < 0 || pad_item > static_cast<int64_t>(INT32_MAX)) {
        MS_LOG(ERROR) << "pad list has invalid num.";
        free(param);
        return nullptr;
      }
    }
    param->pad_u_ = static_cast<int>(*(pad_list->begin()));
    param->pad_d_ = static_cast<int>(*(pad_list->begin() + 1));
    param->pad_l_ = static_cast<int>(*(pad_list->begin() + kOffsetTwo));
    param->pad_r_ = static_cast<int>(*(pad_list->begin() + kOffsetThree));
  }
  param->dilation_h_ = static_cast<int>(*(dilation->begin()));
  param->dilation_w_ = static_cast<int>(*(dilation->begin() + 1));
  param->input_channel_ = static_cast<int>(value->in_channel());
  param->output_channel_ = static_cast<int>(value->out_channel());
  auto pad_mode = value->pad_mode();
  auto act_type = value->activation_type();
  if (SetPadAndAct(pad_mode, act_type, param) != RET_OK) {
    MS_LOG(ERROR) << "SetPadAndAct failed.";
    free(param);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_POPULATE(PrimitiveType_Conv2DFusion, PopulateConvParameter, SCHEMA_CUR)
}  // namespace lite
}  // namespace mindspore
