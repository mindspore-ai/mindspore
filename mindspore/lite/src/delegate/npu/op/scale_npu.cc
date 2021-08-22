/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/delegate/npu/op/scale_npu.h"
#include "src/delegate/npu/npu_converter_utils.h"

namespace mindspore {
constexpr int SCALE_INDEX = 1;
constexpr int BIAS_INDEX = 2;

int ScaleNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                          const std::vector<mindspore::MSTensor> &out_tensors) {
  auto scale_prim = primitive->value_as_ScaleFusion();
  if (scale_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  axis_ = scale_prim->axis();
  if (axis_ < 0) {
    axis_ = axis_ + in_tensors[0].Shape().size();
  }
  if (axis_ != NHWC_C && axis_ != NCHW_C) {
    MS_LOG(WARNING) << "Npu scale axis attr only support 1 or channel, now is " << axis_;
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int ScaleNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                     const std::vector<mindspore::MSTensor> &out_tensors) {
  op_ = new (std::nothrow) hiai::op::Scale(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  op_->set_attr_axis(1);  // only support axis 1 now

  auto scale_prim = primitive->value_as_ScaleFusion();
  if (scale_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  act_type_ = scale_prim->activation_type();
  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    auto ret = SetActivation(op_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return ret;
    }
  }
  return RET_OK;
}

int ScaleNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors,
                             const std::vector<ge::Operator *> &npu_inputs) {
  op_->set_input_x(*npu_inputs.at(0));
  MS_ASSERT(in_tensors.size() > SCALE_INDEX);
  auto scale_shape = in_tensors[SCALE_INDEX].Shape();
  auto scale_tensor = ConverterToNPUTensor(in_tensors[SCALE_INDEX]);
  if (scale_tensor == nullptr) {
    MS_LOG(ERROR) << "Get scale_tensor failed.";
    return RET_ERROR;
  }
  scale_tensor->SetTensorDesc(ge::TensorDesc(ConverterToNPUShape({1, scale_shape[0], 1, 1})));

  scale_ = new (std::nothrow) hiai::op::Const(name_ + "_scale");
  if (scale_ == nullptr) {
    MS_LOG(ERROR) << "New scale_ const failed.";
    return RET_ERROR;
  }
  scale_->set_attr_value(scale_tensor);
  op_->set_input_scale(*scale_);

  if (in_tensors.size() > BIAS_INDEX && in_tensors[BIAS_INDEX] != nullptr) {
    auto bias_shape = in_tensors[BIAS_INDEX].Shape();
    auto bias_tensor = ConverterToNPUTensor(in_tensors[BIAS_INDEX]);
    if (bias_tensor == nullptr) {
      MS_LOG(ERROR) << "Get bias_tensor failed.";
      return RET_ERROR;
    }
    scale_tensor->SetTensorDesc(ge::TensorDesc(ConverterToNPUShape({1, bias_shape[0], 1, 1})));

    bias_ = new (std::nothrow) hiai::op::Const(name_ + "_beta");
    if (bias_ == nullptr) {
      MS_LOG(ERROR) << "New beta_ const failed.";
      return RET_ERROR;
    }
    bias_->set_attr_value(bias_tensor);
    op_->set_input_bias(*bias_);
  }
  return RET_OK;
}

ge::Operator *ScaleNPUOp::GetNPUOp() {
  if (act_type_ == schema::ActivationType_NO_ACTIVATION) {
    return op_;
  } else {
    return act_;
  }
}

int ScaleNPUOp::SetActivation(const ge::Operator *input) {
  act_ = new (std::nothrow) hiai::op::Activation(name_ + "_act");
  if (act_ == nullptr) {
    MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  act_->set_input_x(*input);

  auto act_mode = ConverterToNPUActivationMode(act_type_);
  if (act_mode == ACTIVATION_INVALID) {
    MS_LOG(ERROR) << "Unsupported activation type for scale op " << name_;
    return RET_ERROR;
  }
  act_->set_attr_mode(act_mode);
  return RET_OK;
}

ScaleNPUOp::~ScaleNPUOp() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
  if (scale_ != nullptr) {
    delete scale_;
    scale_ = nullptr;
  }
  if (bias_ != nullptr) {
    delete bias_;
    bias_ = nullptr;
  }
  if (act_ != nullptr) {
    delete act_;
    act_ = nullptr;
  }
}
}  // namespace mindspore
