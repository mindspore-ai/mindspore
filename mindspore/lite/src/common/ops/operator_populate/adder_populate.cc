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
#include "ops/adder.h"
#include "ops/fusion/adder_fusion.h"
using mindspore::ops::kActivationType;
using mindspore::ops::kNameAdder;
using mindspore::ops::kNameAdderFusion;
using mindspore::schema::PrimitiveType_AdderFusion;
namespace mindspore {
namespace lite {
OpParameter *PopulateAdderOpParameter(const BaseOperatorPtr &base_operator) {
  auto param = reinterpret_cast<ConvParameter *>(PopulateOpParameter<ConvParameter>(base_operator));
  if (param == nullptr) {
    MS_LOG(ERROR) << "new ConvParameter failed.";
    return nullptr;
  }
  auto op = dynamic_cast<ops::Adder *>(base_operator.get());
  if (op == nullptr) {
    MS_LOG(ERROR) << "operator is not Adder.";
    free(param);
    return nullptr;
  }
  auto kernel_size = op->get_kernel_size();
  auto stride = op->get_stride();
  auto pad_list = op->get_pad_list();
  auto dilation = op->get_dilation();
  if (kernel_size.size() < kMinShapeSizeTwo || stride.size() < kMinShapeSizeTwo ||
      pad_list.size() < kMinShapeSizeFour || dilation.size() < kMinShapeSizeTwo) {
    MS_LOG(ERROR) << "exist attr size is invalid.";
    free(param);
    return nullptr;
  }
  param->kernel_h_ = static_cast<int>(*(kernel_size.begin()));
  param->kernel_w_ = static_cast<int>(*(kernel_size.begin() + 1));
  param->stride_h_ = static_cast<int>(*(stride.begin()));
  param->stride_w_ = static_cast<int>(*(stride.begin() + 1));
  param->pad_u_ = static_cast<int>(*(pad_list.begin()));
  param->pad_d_ = static_cast<int>(*(pad_list.begin() + 1));
  param->pad_l_ = static_cast<int>(*(pad_list.begin() + kOffsetTwo));
  param->pad_r_ = static_cast<int>(*(pad_list.begin() + kOffsetThree));
  param->dilation_h_ = static_cast<int>(*(dilation.begin()));
  param->dilation_w_ = static_cast<int>(*(dilation.begin() + 1));

  auto attr_act_type = base_operator->GetPrim()->GetAttr(kActivationType);
  if (attr_act_type != nullptr) {
    auto act_type = static_cast<ActType>(GetValue<int64_t>(attr_act_type));
    if (act_type == ActType_Relu || act_type == ActType_Relu6) {
      param->act_type_ = act_type;
    } else {
      param->act_type_ = ActType_No;
    }
  }
  param->output_channel_ = static_cast<int>(op->get_out_channel());
  param->input_channel_ = static_cast<int>(op->get_in_channel());
  param->group_ = static_cast<int>(op->get_group());
  return reinterpret_cast<OpParameter *>(param);
}

REG_OPERATOR_POPULATE(kNameAdder, PrimitiveType_AdderFusion, PopulateAdderOpParameter)
REG_OPERATOR_POPULATE(kNameAdderFusion, PrimitiveType_AdderFusion, PopulateAdderOpParameter)
}  // namespace lite
}  // namespace mindspore
