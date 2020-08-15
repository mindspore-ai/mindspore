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

#include <vector>
#include "src/runtime/kernel/arm/fp32/gather.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Gather;

namespace mindspore::kernel {

int GatherCPUKernel::Init() {
  axis_ = (reinterpret_cast<GatherParameter *>(op_parameter_))->axis_;
  batchDims_ = (reinterpret_cast<GatherParameter *>(op_parameter_))->batchDims_;
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int GatherCPUKernel::ReSize() { return RET_OK; }

int GatherCPUKernel::DoGather(int task_id) {
  auto input_tensor = in_tensors_.at(0);
  auto indices_tensor = in_tensors_.at(1);
  auto out_tensor = out_tensors_.at(0);

  auto input_ptr = reinterpret_cast<float *>(input_tensor->Data());
  auto indices_ptr = reinterpret_cast<int *>(indices_tensor->Data());
  auto output_ptr = reinterpret_cast<float *>(out_tensor->Data());

  auto in_shape = input_tensor->shape();
  int in_rank = in_shape.size();
  int indices_element_size = indices_tensor->ElementsNum();

  const int limit = in_shape[axis_];
  for (size_t i = 0; i < indices_element_size; ++i) {
    if (indices_ptr[i] >= limit) {
      MS_LOG(ERROR) << " indice data: " << indices_ptr[i] << " is not in [ 0, " << limit - 1 << " ]";
      return RET_ERROR;
    }
  }

  int outer_size = 1;
  for (int i = 0; i < axis_; ++i) {
    outer_size *= in_shape[i];
  }

  int inner_size = 1;
  for (int i = axis_ + 1; i < in_rank; ++i) {
    inner_size *= in_shape[i];
  }

  int stride = UP_DIV(outer_size, thread_count_);
  int count = MSMIN(stride, outer_size - stride * task_id);

  input_ptr += stride * task_id * limit;
  output_ptr += stride * task_id * indices_element_size;

  auto error_code = Gather(input_ptr, count, inner_size, limit, indices_ptr, indices_element_size, output_ptr);
  if (error_code != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int GatherRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto gather_kernel = reinterpret_cast<GatherCPUKernel *>(cdata);
  auto error_code = gather_kernel->DoGather(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "GatherRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int GatherCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  int error_code = LiteBackendParallelLaunch(GatherRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Gather function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuGatherFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx,
                                               const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Gather);

  auto *kernel = new (std::nothrow) GatherCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Gather, CpuGatherFp32KernelCreator)
}  // namespace mindspore::kernel
