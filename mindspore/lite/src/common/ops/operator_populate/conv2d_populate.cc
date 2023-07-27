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
#include "src/common/ops/operator_populate/utils.h"
#include "nnacl/conv_parameter.h"
#include "ops/conv2d.h"
#include "ops/fusion/conv2d_fusion.h"
using mindspore::ops::kActivationType;
using mindspore::ops::kInChannel;
using mindspore::ops::kNameConv2D;
using mindspore::ops::kNameConv2DFusion;
using mindspore::ops::kPadList;
using mindspore::schema::PrimitiveType_Conv2DFusion;
namespace mindspore {
namespace lite {
namespace {
int SetPadModeAndActType(schema::PadMode pad_mode, schema::ActivationType act_type, ConvParameter *param) {
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

OpParameter *PopulateConv2dOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ConvParameter *>(PopulateOpParameter<ConvParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ConvParameter failed.";
    return nullptr;
  }

  auto op = dynamic_cast<ops::Conv2D *>(base_operator.get());
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
  param->kernel_h_ = static_cast<int>(*(kernel_size.begin()));
  param->kernel_w_ = static_cast<int>(*(kernel_size.begin() + 1));

  auto stride = op->get_stride();
  if (stride.size() < kMinShapeSizeTwo) {
    MS_LOG(ERROR) << " Invalid stride size: " << stride.size();
    free(param);
    return nullptr;
  }
  for (size_t i = 0; i <= 1; i++) {
    auto stride_item = *(stride.begin() + i);
    if (stride_item < 0 || stride_item > static_cast<int64_t>(INT32_MAX)) {
      MS_LOG(ERROR) << "strides has invalid num.";
      free(param);
      return nullptr;
    }
  }
  param->stride_h_ = static_cast<int>(*(stride.begin()));
  param->stride_w_ = static_cast<int>(*(stride.begin() + 1));

  auto pad_list = GetAttrWithDefault<std::vector<int64_t>>(base_operator, kPadList, {0, 0, 0, 0});
  if (pad_list.size() < kMinShapeSizeFour) {
    param->pad_u_ = 0;
    param->pad_d_ = 0;
    param->pad_l_ = 0;
    param->pad_r_ = 0;
  } else {
    for (size_t i = 0; i <= kOffsetThree; i++) {
      auto pad_item = *(pad_list.begin() + i);
      if (pad_item < 0 || pad_item > static_cast<int64_t>(INT32_MAX)) {
        MS_LOG(ERROR) << "pad list has invalid num.";
        free(param);
        return nullptr;
      }
    }
    param->pad_u_ = static_cast<int>(*(pad_list.begin()));
    param->pad_d_ = static_cast<int>(*(pad_list.begin() + 1));
    param->pad_l_ = static_cast<int>(*(pad_list.begin() + kOffsetTwo));
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

  auto in_channel = GetAttrWithDefault<int64_t>(base_operator, kInChannel, 0);
  CHECK_LESS_RETURN_RET(INT32_MAX, in_channel, nullptr, param);
  param->input_channel_ = static_cast<int>(in_channel);

  param->output_channel_ = static_cast<int>(op->get_out_channel());
  param->group_ = static_cast<int>(op->get_group());

  auto act_type = static_cast<schema::ActivationType>(
    GetAttrWithDefault<int64_t>(base_operator, kActivationType, schema::ActivationType_NO_ACTIVATION));
  auto pad_mode = static_cast<schema::PadMode>(op->get_pad_mode());
  if (SetPadModeAndActType(pad_mode, act_type, param) != RET_OK) {
    MS_LOG(ERROR) << "SetPadModeAndActType failed.";
    free(param);
    return nullptr;
  }
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameConv2D, PrimitiveType_Conv2DFusion, PopulateConv2dOpParameter)
REG_OPERATOR_POPULATE(kNameConv2DFusion, PrimitiveType_Conv2DFusion, PopulateConv2dOpParameter)
}  // namespace lite
}  // namespace mindspore
