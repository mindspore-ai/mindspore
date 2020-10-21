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
#include "src/runtime/kernel/arm/fp16/stack_fp16.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/stack_parameter.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/fp16/stack_fp16.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Stack;

namespace mindspore::kernel {

int StackFp16CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

void StackFp16CPUKernel::InitMallocFlags() {
  malloc_buffers_.resize(in_tensors_.size());
  for (size_t i = 0; i < in_tensors_.size(); ++i) {
    malloc_buffers_[i] = in_tensors_[i]->data_type() == kNumberTypeFloat32;
  }
  malloc_out = out_tensors_[0]->data_type() == kNumberTypeFloat32;
}

int StackFp16CPUKernel::MallocAssignBuffer() {
  buffers_.resize(in_tensors_.size(), nullptr);
  for (size_t i = 0; i < in_tensors_.size(); ++i) {
    buffers_[i] = ConvertInputFp32toFp16(in_tensors_[i], context_);
    if (buffers_[i] == nullptr) {
      return RET_ERROR;
    }
  }

  out_buffer_ = nullptr;
  out_buffer_ = MallocOutputFp16(out_tensors_[0], context_);
  if (out_buffer_ == nullptr) {
    return RET_ERROR;
  }
  return RET_OK;
}

void StackFp16CPUKernel::FreeBuffer() {
  for (size_t i = 0; i < buffers_.size(); ++i) {
    if (malloc_buffers_[i] && buffers_[i] != nullptr) {
      context_->allocator->Free(buffers_[i]);
      buffers_[i] = nullptr;
    }
  }
  if (malloc_out && out_buffer_ != nullptr) {
    context_->allocator->Free(out_buffer_);
    out_buffer_ = nullptr;
  }
}

int StackFp16CPUKernel::Run() {
  size_t inputs_num = in_tensors_.size();
  auto input0 = in_tensors_[0];
  if (inputs_num == 1) {
    memcpy(out_tensors_[0]->MutableData(), input0->MutableData(), input0->Size());
    return RET_OK;
  }
  InitMallocFlags();
  auto ret = MallocAssignBuffer();
  if (ret != RET_OK) {
    FreeBuffer();
    return ret;
  }
  auto input0_shape = input0->shape();
  DoStackFp16(buffers_.data(), inputs_num, input0_shape.data(), input0_shape.size(), axis_, out_buffer_);
  // if output tensor is fp32, we need to transform
  if (malloc_out) {
    auto out_tensor = out_tensors_.at(0);
    Float16ToFloat32(out_buffer_, reinterpret_cast<float *>(out_tensor->MutableData()), out_tensor->ElementsNum());
  }

  FreeBuffer();
  return RET_OK;
}

kernel::LiteKernel *CpuStackFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                              const std::vector<lite::Tensor *> &outputs, OpParameter *op_parameter,
                                              const lite::InnerContext *ctx, const kernel::KernelKey &desc,
                                              const mindspore::lite::PrimitiveC *primitive) {
  if (op_parameter == nullptr) {
    MS_LOG(ERROR) << "Input op_parameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Stack);
  auto *kernel = new (std::nothrow) StackFp16CPUKernel(op_parameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new StackFp16CPUKernel fail!";
    free(op_parameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << op_parameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Stack, CpuStackFp16KernelCreator)
}  // namespace mindspore::kernel
