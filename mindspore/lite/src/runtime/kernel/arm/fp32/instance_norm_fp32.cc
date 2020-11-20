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
#include "src/runtime/kernel/arm/fp32/instance_norm_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
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
  auto input_shapes = in_tensors_.front()->shape();
  auto n_dim = input_shapes.size();
  outer_size_ = input_shapes[0] * input_shapes[n_dim - 1];
  inner_size_ = 1;
  for (size_t i = 0; i < n_dim - 1; ++i) {
    inner_size_ *= input_shapes[i];
  }
  param_->channel_ = input_shapes[n_dim - 1];
  return RET_OK;
}

int InstanceNormCPUKernel::DoInstanceNorm(int task_id) {
  int ret = InstanceNorm(outer_size_, inner_size_, src_data_, scale_data_, bias_data_, param_, dst_data_, task_id,
                         op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoInstanceNorm error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int InstanceNormRun(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<InstanceNormCPUKernel *>(cdata);
  auto ret = kernel->DoInstanceNorm(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "InstanceNormRun error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int InstanceNormCPUKernel::Run() {
  src_data_ = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  scale_data_ = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  bias_data_ = reinterpret_cast<float *>(in_tensors_.at(2)->MutableData());
  dst_data_ = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  auto ret = ParallelLaunch(this->context_->thread_pool_, InstanceNormRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FillRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuInstanceNormFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                     const std::vector<lite::Tensor *> &outputs,
                                                     OpParameter *opParameter, const lite::InnerContext *ctx,
                                                     const kernel::KernelKey &desc,
                                                     const mindspore::lite::PrimitiveC *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, opParameter is nullptr, type: PrimitiveType_InstanceNorm. ";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_InstanceNorm);
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_InstanceNorm, CpuInstanceNormFp32KernelCreator)
}  // namespace mindspore::kernel
