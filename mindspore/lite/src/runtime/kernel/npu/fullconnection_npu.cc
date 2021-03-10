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

#include "src/runtime/kernel/npu/fullconnection_npu.h"
#include <memory>
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"
using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_FullConnection;

namespace mindspore::kernel {
int FullconnectionNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs,
                                       const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter) {
  return RET_OK;
}

int FullconnectionNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                          const std::vector<lite::Tensor *> &outputs,
                                          const std::vector<ge::Operator *> &npu_inputs) {
  auto input_shape = inputs[0]->shape();
  reshape_ = new (std::nothrow) hiai::op::Reshape(name_ + "_reshape");
  if (reshape_ == nullptr) {
    MS_LOG(ERROR) << "New reshape operator for fullconnection op " << name_ << " failed.";
    return RET_ERROR;
  }
  reshape_->set_input_x(*npu_inputs[0]);
  int col = 1;
  for (int i = 1; i < input_shape.size(); i++) {
    col *= input_shape[i];
  }
  reshape_op_ = new (std::nothrow) hiai::op::Const(name_ + "_reshape_data");
  vector<int> reshape_data = {input_shape[0], col};
  ge::TensorDesc reshape_tensor_desc(ge::Shape({2}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorPtr reshape_tensor = std::make_shared<hiai::Tensor>(reshape_tensor_desc);
  reshape_tensor->SetData(reinterpret_cast<uint8_t *>(reshape_data.data()), 2 * sizeof(float));
  reshape_op_->set_attr_value(reshape_tensor);
  reshape_->set_input_shape(*reshape_op_);

  fc_ = new (std::nothrow) hiai::op::MatMul(name_);
  if (fc_ == nullptr) {
    MS_LOG(ERROR) << "New matmul operator for fullconnection op " << name_ << " failed.";
    return RET_ERROR;
  }
  fc_->set_input_x1(*reshape_);

  weight_ = new (std::nothrow) hiai::op::Const(name_ + "_w");
  if (weight_ == nullptr) {
    MS_LOG(ERROR) << "New weight const failed.";
    return RET_ERROR;
  }
  inputs[1]->set_format(schema::Format_NCHW);
  auto weight_tensor = mindspore::lite::ConverterToNPUTensor(inputs[1]);
  weight_->set_attr_value(weight_tensor);
  inputs[1]->set_format(schema::Format_NHWC);
  fc_->set_input_x2(*weight_).set_attr_transpose_x2(true);

  if (fc_param_->has_bias_) {
    biasadd_ = new (std::nothrow) hiai::op::BiasAdd(name_ + "_biasadd");
    if (biasadd_ == nullptr) {
      MS_LOG(ERROR) << "New biasadd operator for fullconnection op " << name_ << " failed.";
      return RET_ERROR;
    }

    auto ret = InitBiasConst(inputs);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "Set bias for convolution op " << name_ << " failed when running npu";
      return RET_ERROR;
    }
    biasadd_->set_input_x(*fc_).set_input_bias(*bias_);
  }

  if (fc_param_->act_type_ != ActType_No) {
    auto ret =
      biasadd_ == nullptr ? SetActivation(fc_, fc_param_->act_type_) : SetActivation(biasadd_, fc_param_->act_type_);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

ge::Operator *mindspore::kernel::FullconnectionNPUKernel::GetNPUOp() {
  if (fc_param_->act_type_ != ActType_No) {
    return act_;
  }
  if (fc_param_->has_bias_) {
    return biasadd_;
  }
  return fc_;
}

FullconnectionNPUKernel::~FullconnectionNPUKernel() {
  if (reshape_ != nullptr) {
    delete reshape_;
    reshape_ = nullptr;
  }
  if (fc_ != nullptr) {
    delete fc_;
    fc_ = nullptr;
  }
  if (biasadd_ != nullptr) {
    delete biasadd_;
    biasadd_ = nullptr;
  }
  if (reshape_op_ != nullptr) {
    delete reshape_op_;
    reshape_op_ = nullptr;
  }
}
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_FullConnection, NPUKernelCreator<FullconnectionNPUKernel>)
}  // namespace mindspore::kernel
