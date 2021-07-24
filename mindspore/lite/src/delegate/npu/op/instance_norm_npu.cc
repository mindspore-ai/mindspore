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

#include "src/delegate/npu/op/instance_norm_npu.h"
#include "src/delegate/npu/npu_converter_utils.h"

namespace mindspore {
constexpr int GAMMA_INDEX = 1;
constexpr int BETA_INDEX = 2;

int InstanceNormNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                            const std::vector<mindspore::MSTensor> &out_tensors) {
  instance_norm_ = new (std::nothrow) hiai::op::InstanceNorm(name_);
  if (instance_norm_ == nullptr) {
    MS_LOG(ERROR) << "New instance norm npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  auto instance_norm_prim = primitive->value_as_InstanceNorm();
  if (instance_norm_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  instance_norm_->set_attr_epsilon(instance_norm_prim->epsilon());
  return RET_OK;
}

int InstanceNormNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                                    const std::vector<mindspore::MSTensor> &out_tensors,
                                    const std::vector<ge::Operator *> &npu_inputs) {
  instance_norm_->set_input_x(*npu_inputs[0]);

  auto gamma_shape = in_tensors[GAMMA_INDEX].Shape();
  auto gamma_tensor = ConverterToNPUTensor(in_tensors[GAMMA_INDEX]);
  if (gamma_tensor == nullptr) {
    MS_LOG(ERROR) << "Get gamma_tensor failed.";
    return RET_ERROR;
  }
  gamma_tensor->SetTensorDesc(ge::TensorDesc(ConverterToNPUShape({1, gamma_shape[0], 1, 1})));

  gamma_ = new (std::nothrow) hiai::op::Const(name_ + "_gamma");
  if (gamma_ == nullptr) {
    MS_LOG(ERROR) << "New gamma_ const failed.";
    return RET_ERROR;
  }
  gamma_->set_attr_value(gamma_tensor);
  instance_norm_->set_input_gamma(*gamma_);

  auto beta_shape = in_tensors[BETA_INDEX].Shape();
  auto beta_tensor = ConverterToNPUTensor(in_tensors[BETA_INDEX]);
  if (beta_tensor == nullptr) {
    MS_LOG(ERROR) << "Get beta_tensor failed.";
    return RET_ERROR;
  }
  beta_tensor->SetTensorDesc(ge::TensorDesc(ConverterToNPUShape({1, beta_shape[0], 1, 1})));

  beta_ = new (std::nothrow) hiai::op::Const(name_ + "_beta");
  if (beta_ == nullptr) {
    MS_LOG(ERROR) << "New beta_ const failed.";
    return RET_ERROR;
  }
  beta_->set_attr_value(beta_tensor);
  instance_norm_->set_input_beta(*beta_);
  return RET_OK;
}

ge::Operator *InstanceNormNPUOp::GetNPUOp() { return this->instance_norm_; }

InstanceNormNPUOp::~InstanceNormNPUOp() {
  if (instance_norm_ != nullptr) {
    delete instance_norm_;
    instance_norm_ = nullptr;
  }
  if (gamma_ != nullptr) {
    delete gamma_;
    gamma_ = nullptr;
  }
  if (beta_ != nullptr) {
    delete beta_;
    beta_ = nullptr;
  }
}
}  // namespace mindspore
