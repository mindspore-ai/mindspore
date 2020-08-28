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
#include "src/runtime/kernel/arm/fp32/gather.h"
#include <vector>
#include "nnacl/gather_parameter.h"
#include "nnacl/fp32/gather.h"
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
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

GatherCPUKernel::~GatherCPUKernel() {
  context_->allocator->Free(indices_data_);
}

int GatherCPUKernel::ReSize() { return RET_OK; }

int GatherCPUKernel::DoGather(int task_id) {
  auto input_tensor = in_tensors_.at(0);
  auto indices_tensor = in_tensors_.at(1);
  auto out_tensor = out_tensors_.at(0);

  auto input_ptr = reinterpret_cast<float *>(input_tensor->Data());
  auto output_ptr = reinterpret_cast<float *>(out_tensor->Data());

  auto input_int32 = reinterpret_cast<int32_t *>(input_tensor->Data());
  auto output_int32 = reinterpret_cast<int32_t *>(out_tensor->Data());

  auto in_shape = input_tensor->shape();
  int in_rank = in_shape.size();
  int indices_element_size = indices_tensor->ElementsNum();
  auto axis = (reinterpret_cast<GatherParameter *>(op_parameter_))->axis_;

  const int limit = in_shape[axis];

  int outer_size = 1, inner_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= in_shape[i];
  }
  for (int i = axis + 1; i < in_rank; ++i) {
    inner_size *= in_shape[i];
  }
  int stride = UP_DIV(outer_size, op_parameter_->thread_num_);
  int count = MSMIN(stride, outer_size - stride * task_id);
  auto thread_stride = stride * task_id;

  int error_code;
  if (input_tensor->data_type() == kNumberTypeInt32) {
    input_int32 += thread_stride * limit;
    output_int32 += thread_stride * indices_element_size;
    error_code = GatherInt32(input_int32, count, inner_size, limit, indices_data_, indices_element_size, output_int32);
  } else {
    input_ptr += thread_stride * limit;
    output_ptr += thread_stride * indices_element_size;
    error_code = Gather(input_ptr, count, inner_size, limit, indices_data_, indices_element_size, output_ptr);
  }
  return error_code;
}

int GatherRun(void *cdata, int task_id) {
  auto gather_kernel = reinterpret_cast<GatherCPUKernel *>(cdata);
  auto error_code = gather_kernel->DoGather(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "GatherRun error task_id[" << task_id << "] error_code[" << error_code << "]";
  }
  return error_code;
}

int GatherCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }

  auto indices_tensor = in_tensors_.at(1);
  indices_data_ = reinterpret_cast<int *>(context_->allocator->Malloc(indices_tensor->Size()));
  if (indices_data_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    return RET_ERROR;
  }
  auto in_shape = in_tensors_.at(0)->shape();
  int indices_element_size = indices_tensor->ElementsNum();
  auto axis = (reinterpret_cast<GatherParameter *>(op_parameter_))->axis_;;
  auto indices_ptr = reinterpret_cast<float *>(indices_tensor->Data());
  const int limit = in_shape[axis];
  for (int i = 0; i < indices_element_size; ++i) {
    indices_data_[i] = static_cast<int>(indices_ptr[i]);
    if (indices_data_[i] >= limit) {
      MS_LOG(ERROR) << " indice data: " << indices_data_[i] << " is not in [ 0, " << limit - 1 << " ]";
      return RET_ERROR;
    }
  }
  int error_code = ParallelLaunch(THREAD_POOL_DEFAULT, GatherRun, this, op_parameter_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Gather function error error_code[" << error_code << "]";
  }
  return error_code;
}

kernel::LiteKernel *CpuGatherFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                               const std::vector<lite::tensor::Tensor *> &outputs,
                                               OpParameter *opParameter, const lite::Context *ctx,
                                               const kernel::KernelKey &desc,
                                               const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(desc.type == schema::PrimitiveType_Gather);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr!";
    return nullptr;
  }
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
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Gather, CpuGatherFp32KernelCreator)
}  // namespace mindspore::kernel
