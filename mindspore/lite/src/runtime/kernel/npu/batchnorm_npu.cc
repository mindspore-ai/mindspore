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

#include "src/runtime/kernel/npu/batchnorm_npu.h"
#include "include/graph/op/all_ops.h"
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_FusedBatchNorm;

namespace mindspore::kernel {
int BatchnormNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                  OpParameter *opParameter) {
  return RET_OK;
}

int BatchnormNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                     const std::vector<lite::Tensor *> &outputs,
                                     const std::vector<ge::Operator *> &npu_inputs) {
  batchnorm_ = new (std::nothrow) ge::op::BatchNormExt2(name_);
  if (batchnorm_ == nullptr) {
    MS_LOG(ERROR) << "New batchnorm npu operator for batchnorm op " << name_ << " failed.";
    return RET_ERROR;
  }
  batchnorm_->set_input_x(*npu_inputs[0]);

  auto scale = new (std::nothrow) hiai::op::Const(name_ + "_scale");
  if (scale == nullptr) {
    MS_LOG(ERROR) << "New scale const failed.";
    return RET_ERROR;
  }
  auto scale_tensor = mindspore::lite::ConverterToNPUTensor(inputs[1]);
  scale->set_attr_value(scale_tensor);
  batchnorm_->set_input_scale(*scale);

  auto offset = new (std::nothrow) hiai::op::Const(name_ + "_offset");
  if (offset == nullptr) {
    MS_LOG(ERROR) << "New offset const failed.";
    return RET_ERROR;
  }
  auto offset_tensor = mindspore::lite::ConverterToNPUTensor(inputs[1]);
  offset->set_attr_value(offset_tensor);
  batchnorm_->set_input_offset(*offset);

  auto mean = new (std::nothrow) hiai::op::Const(name_ + "_mean");
  if (mean == nullptr) {
    MS_LOG(ERROR) << "New mean const failed.";
    return RET_ERROR;
  }
  auto mean_tensor = mindspore::lite::ConverterToNPUTensor(inputs[1]);
  mean->set_attr_value(mean_tensor);
  batchnorm_->set_input_mean(*mean);

  auto variance = new (std::nothrow) hiai::op::Const(name_ + "_variance");
  if (variance == nullptr) {
    MS_LOG(ERROR) << "New variance const failed.";
    return RET_ERROR;
  }
  auto variance_tensor = mindspore::lite::ConverterToNPUTensor(inputs[1]);
  variance->set_attr_value(variance_tensor);
  batchnorm_->set_input_variance(*variance);

  batchnorm_->set_attr_epsilon(batchnorm_param_->epsilon_);
  batchnorm_->set_attr_momentum(batchnorm_param_->momentum_);
  batchnorm_->set_attr_mode(1);
  return RET_OK;
}

ge::Operator *mindspore::kernel::BatchnormNPUKernel::GetNPUOp() { return batchnorm_; }

BatchnormNPUKernel::~BatchnormNPUKernel() {
  if (batchnorm_ != nullptr) {
    delete batchnorm_;
    batchnorm_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_FusedBatchNorm, NPUKernelCreator<BatchnormNPUKernel>)
}  // namespace mindspore::kernel
