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
#include "src/common/ops/operator_populate/operator_populate_register.h"
#include "nnacl/conv_parameter.h"
#include "ops/fusion/conv2d_transpose_fusion.h"
using mindspore::ops::kNameConv2dTransposeFusion;
using mindspore::ops::kPadMode;
using mindspore::schema::PrimitiveType_Conv2dTransposeFusion;
namespace mindspore {
namespace lite {
namespace {
int SetPadModeAndAct(schema::PadMode pad_mode, schema::ActivationType act_type, ConvParameter *param) {
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
}  // namespace

OpParameter *PopulateDeconv2dOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ConvParameter *>(PopulateOpParameter<ConvParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ConvParameter failed.";
    return nullptr;
  }

  auto op = dynamic_cast<ops::Conv2dTransposeFusion *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not Adder.";
    free(param);
    return nullptr;
  }

  auto kernel_size = op->get_kernel_size();
  param->kernel_h_ = -1;
  param->kernel_w_ = -1;
  if (kernel_size.size() < kMinShapeSizeTwo) {
    MS_LOG(ERROR) << "kernel size is invalid.";
    free(param);
    return nullptr;
  }
  CHECK_LESS_RETURN_RET(INT32_MAX, *(kernel_size.begin()), nullptr, param);
  param->kernel_h_ = static_cast<int>(*(kernel_size.begin()));
  CHECK_LESS_RETURN_RET(INT32_MAX, *(kernel_size.begin() + 1), nullptr, param);
  param->kernel_w_ = static_cast<int>(*(kernel_size.begin() + 1));

  auto stride = op->get_stride();
  if (stride.size() < kMinShapeSizeTwo) {
    MS_LOG(ERROR) << " Invalid stride size: " << stride.size();
    free(param);
    return nullptr;
  }
  CHECK_LESS_RETURN_RET(INT32_MAX, *(stride.begin()), nullptr, param);
  param->stride_h_ = static_cast<int>(*(stride.begin()));
  CHECK_LESS_RETURN_RET(INT32_MAX, *(stride.begin() + 1), nullptr, param);
  param->stride_w_ = static_cast<int>(*(stride.begin() + 1));

  auto pad_list = op->get_pad_list();
  if (pad_list.size() < kMinShapeSizeFour) {
    param->pad_u_ = 0;
    param->pad_d_ = 0;
    param->pad_l_ = 0;
    param->pad_r_ = 0;
  } else {
    CHECK_LESS_RETURN_RET(INT32_MAX, *(pad_list.begin()), nullptr, param);
    param->pad_u_ = static_cast<int>(*(pad_list.begin()));
    CHECK_LESS_RETURN_RET(INT32_MAX, *(pad_list.begin() + 1), nullptr, param);
    param->pad_d_ = static_cast<int>(*(pad_list.begin() + 1));
    CHECK_LESS_RETURN_RET(INT32_MAX, *(pad_list.begin() + kOffsetTwo), nullptr, param);
    param->pad_l_ = static_cast<int>(*(pad_list.begin() + kOffsetTwo));
    CHECK_LESS_RETURN_RET(INT32_MAX, *(pad_list.begin() + kOffsetThree), nullptr, param);
    param->pad_r_ = static_cast<int>(*(pad_list.begin() + kOffsetThree));
  }

  auto dilation = op->get_dilation();
  if (dilation.size() < kMinShapeSizeTwo) {
    MS_LOG(ERROR) << "Invalid dilation size: " << dilation.size();
    free(param);
    return nullptr;
  }
  CHECK_LESS_RETURN_RET(INT32_MAX, *(dilation.begin()), nullptr, param);
  param->dilation_h_ = static_cast<int>(*(dilation.begin()));
  CHECK_LESS_RETURN_RET(INT32_MAX, *(dilation.begin() + 1), nullptr, param);
  param->dilation_w_ = static_cast<int>(*(dilation.begin() + 1));

  auto output_paddings = op->get_output_paddings();
  param->output_padding_h_ = 0;
  param->output_padding_w_ = 0;
  if (output_paddings.size() < kMinShapeSizeTwo) {
    MS_LOG(ERROR) << "output_paddings size is invalid.";
    free(param);
    return nullptr;
  }
  CHECK_LESS_RETURN_RET(INT32_MAX, *(output_paddings.begin()), nullptr, param);
  param->output_padding_h_ = static_cast<int>(*(output_paddings.begin()));
  CHECK_LESS_RETURN_RET(INT32_MAX, *(output_paddings.begin() + 1), nullptr, param);
  param->output_padding_w_ = static_cast<int>(*(output_paddings.begin() + 1));
  if (param->output_padding_h_ < 0 || param->output_padding_w_ < 0) {
    MS_LOG(ERROR) << "Invalid output padding";
    free(param);
    return nullptr;
  }

  param->output_channel_ = static_cast<int>(op->get_out_channel());
  param->input_channel_ = static_cast<int>(op->get_in_channel());
  param->group_ = static_cast<int>(op->get_group());
  auto act_type = static_cast<schema::ActivationType>(op->get_activation_type());
  auto attr_pad_mode = base_operator->GetPrim()->GetAttr(kPadMode);
  if (attr_pad_mode == nullptr) {
    MS_LOG(ERROR) << "The attr(" << kPadMode << ") of operator(" << base_operator->name() << ") not exist";
    free(param);
    return nullptr;
  }
  auto pad_mode = static_cast<schema::PadMode>(GetValue<int64_t>(attr_pad_mode));

  if (SetPadModeAndAct(pad_mode, act_type, param) != RET_OK) {
    MS_LOG(ERROR) << "SetPadModeAndActType failed.";
    free(param);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameConv2dTransposeFusion, PrimitiveType_Conv2dTransposeFusion, PopulateDeconv2dOpParameter)
}  // namespace lite
}  // namespace mindspore
