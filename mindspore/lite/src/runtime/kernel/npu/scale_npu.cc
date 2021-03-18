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

#include "src/runtime/kernel/npu/scale_npu.h"
#include <memory>
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::Format_NHWC;
using mindspore::schema::PrimitiveType_ScaleFusion;

namespace mindspore::kernel {
int ScaleNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                              OpParameter *opParameter) {
  if (scale_parameter_->axis_ < 0) {
    scale_parameter_->axis_ = scale_parameter_->axis_ + inputs[0]->shape().size();
  }
  if (inputs.size() > 1 && inputs[0]->shape().size() == 4 && inputs[0]->format() == schema::Format_NHWC) {
    if (scale_parameter_->axis_ != 3) {
      MS_LOG(ERROR) << "Npu scale axis attr only support on channel, now is " << scale_parameter_->axis_;
      return RET_ERROR;
    }
    return RET_OK;
  }
  if (scale_parameter_->axis_ != 1) {
    MS_LOG(ERROR) << "Npu scale axis attr only support 1, now is " << scale_parameter_->axis_;
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                 const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::Scale(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  op_->set_attr_axis(1);  // only support axis 1 now
  op_->set_input_x(*npu_inputs.at(0));

  MS_ASSERT(inputs.size() > 1);
  auto scale_shape = inputs.at(1)->shape();
  std::shared_ptr<ge::Tensor> scale_tensor = std::shared_ptr<ge::Tensor>(new (std::nothrow) ge::Tensor());
  if (scale_tensor == nullptr) {
    MS_LOG(ERROR) << "new scale_tensor failed.";
    return RET_ERROR;
  }
  ge::TensorDesc scale_tensor_desc(lite::ConverterToNPUShape({1, scale_shape[0], 1, 1}), ge::FORMAT_NCHW,
                                   lite::ConverterToNPUDataType(inputs[1]->data_type()));
  scale_tensor->SetTensorDesc(scale_tensor_desc);
  scale_tensor->SetData(reinterpret_cast<const uint8_t *>(inputs[1]->data_c()), inputs[1]->Size());
  scale_ = new (std::nothrow) hiai::op::Const(name_ + "_scale");
  if (scale_ == nullptr) {
    MS_LOG(ERROR) << "New scale_ const failed.";
    return RET_ERROR;
  }
  scale_->set_attr_value(scale_tensor);
  op_->set_input_scale(*scale_);

  if (inputs.size() > 2 && inputs[2] != nullptr) {
    auto bias_shape = inputs[2]->shape();
    std::shared_ptr<ge::Tensor> bias_tensor = std::shared_ptr<ge::Tensor>(new (std::nothrow) ge::Tensor());
    if (bias_tensor == nullptr) {
      MS_LOG(ERROR) << "new bias_tensor failed.";
      return RET_ERROR;
    }
    ge::TensorDesc bias_tensor_desc(lite::ConverterToNPUShape({1, bias_shape[0], 1, 1}), ge::FORMAT_NCHW,
                                    lite::ConverterToNPUDataType(inputs[2]->data_type()));
    bias_tensor->SetTensorDesc(bias_tensor_desc);
    bias_tensor->SetData(reinterpret_cast<const uint8_t *>(inputs[2]->data_c()), inputs[2]->Size());
    bias_ = new (std::nothrow) hiai::op::Const(name_ + "_beta");
    if (bias_ == nullptr) {
      MS_LOG(ERROR) << "New beta_ const failed.";
      return RET_ERROR;
    }
    bias_->set_attr_value(bias_tensor);
    op_->set_input_bias(*bias_);
  }

  if (scale_parameter_->activation_type_ != schema::ActivationType_NO_ACTIVATION) {
    auto ret = SetActivation(op_, scale_parameter_->activation_type_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return ret;
    }
  }

  return RET_OK;
}

ge::Operator *mindspore::kernel::ScaleNPUKernel::GetNPUOp() {
  if (scale_parameter_->activation_type_ == schema::ActivationType_NO_ACTIVATION) {
    return op_;
  } else {
    return act_;
  }
}

int ScaleNPUKernel::SetActivation(const ge::Operator *input, int act_type) {
  act_ = new (std::nothrow) hiai::op::Activation(name_ + "_act");
  if (act_ == nullptr) {
    MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  act_->set_input_x(*input);
  if (act_type == schema::ActivationType_RELU) {
    act_->set_attr_mode(1);
  } else if (act_type == schema::ActivationType_RELU6) {
    act_->set_attr_mode(14);
  } else {
    MS_LOG(ERROR) << "Unsupported activation type for scale.";
    return RET_ERROR;
  }
  return RET_OK;
}

ScaleNPUKernel::~ScaleNPUKernel() {
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

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_ScaleFusion, NPUKernelCreator<ScaleNPUKernel>)
}  // namespace mindspore::kernel
