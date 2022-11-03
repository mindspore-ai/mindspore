/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/groupnorm_fp32.h"
#include "src/litert/kernel_registry.h"
#include "src/common/tensor_util.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_GroupNormFusion;
namespace {}  // namespace
namespace mindspore::kernel {
GroupnormCPUKernel::GroupnormCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                       const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
    : LiteKernel(parameter, inputs, outputs, ctx) {
  if (in_tensors_.size() != DIMENSION_3D) {
    return;
  }
  if (out_tensors_.size() != 1) {
    return;
  }

  for (size_t i = 0; i < in_tensors_.size(); i++) {
    (void)Tensor2TensorC(in_tensors_.at(i), &(in_[i]));
  }
  (void)Tensor2TensorC(out_tensors_.at(0), &(out_[0]));
}

GroupnormCPUKernel::~GroupnormCPUKernel() {
  if (kernel_ == nullptr) {
    return;
  }
  kernel_->release(kernel_);
  free(kernel_);
}

int GroupnormCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_3D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (!InferShapeDone()) {
    return RET_OK;
  }
  kernel_ =
    CreateKernel(op_parameter_, in_, in_tensors().size(), out_, out_tensors_.size(), kNumberTypeFloat32, Format_NCHW);
  if (kernel_ == nullptr) {
    return RET_NULL_PTR;
  }
  return ReSize();
}

int GroupnormCPUKernel::ReSize() {
  if (kernel_ == nullptr) {
    return RET_ERROR;
  }
  return kernel_->resize(kernel_);
}

int GroupnormCPUKernel::Run() {
  if (kernel_ == nullptr) {
    return RET_ERROR;
  }
  kernel_->in[0].data_ = in_tensors().at(0)->data();
  kernel_->in[C1NUM].data_ = in_tensors().at(C1NUM)->data();
  kernel_->in[C2NUM].data_ = in_tensors().at(C2NUM)->data();
  kernel_->out[0].data_ = out_tensors().front()->data();
  return kernel_->compute(kernel_);
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_GroupNormFusion, LiteKernelCreator<GroupnormCPUKernel>)
}  // namespace mindspore::kernel
