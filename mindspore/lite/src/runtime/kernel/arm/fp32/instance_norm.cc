/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp32/instance_norm.h"
#include "nnacl/fp32/instance_norm.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_InstanceNorm;

namespace mindspore::kernel {
int InstanceNormCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int InstanceNormCPUKernel::ReSize() {
  auto input_shapes = in_tensors_[0]->shape();
  auto n_dim = input_shapes.size();
  auto param = reinterpret_cast<InstanceNormParameter *>(op_parameter_);
  param->batch_ = input_shapes[0];
  param->channel_ = input_shapes[n_dim - 1];
  param->unit_ = 1;
  for (size_t i = 1; i < n_dim - 1; i++) {
    param->unit_ *= input_shapes[i];
  }
  return RET_OK;
}

int InstanceNormCPUKernel::Run() {
  auto ret = ParallelLaunch(this->context_->thread_pool_, InstanceNormRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InstanceNormRun error error_code[" << ret << "]";
  }
  return ret;
}

int InstanceNormCPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<InstanceNormParameter *>(op_parameter_);
  InstanceNormFp32(in_tensors_.at(0)->MutableData(), in_tensors_.at(1)->MutableData(), in_tensors_.at(2)->MutableData(),
                   param, task_id, out_tensors_.at(0)->MutableData());
  return mindspore::lite::RET_OK;
}

int InstanceNormRun(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<InstanceNormCPUKernel *>(cdata);
  auto ret = kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InstanceNormRun error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

kernel::LiteKernel *CpuInstanceNormKernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                 const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                 const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                                 const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  auto *kernel = new (std::nothrow) InstanceNormCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new InstanceNormCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_InstanceNorm, CpuInstanceNormKernelCreator)
}  // namespace mindspore::kernel
