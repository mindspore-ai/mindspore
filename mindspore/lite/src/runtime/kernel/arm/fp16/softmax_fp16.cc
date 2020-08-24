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

#include <string.h>
#include <vector>
#include "src/runtime/kernel/arm/fp16/softmax_fp16.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "nnacl/fp16/softmax_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SoftMax;

namespace mindspore::kernel {
int SoftmaxFp16CPUKernel::Init() {
  auto ret = SoftmaxBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SoftmaxFp16CPUKernel::ReSize() {
  return SoftmaxBaseCPUKernel::ReSize();
}

int SoftmaxFp16CPUKernel::MallocTmpBuffer() {
  auto n_dim = softmax_param_->n_dim_;
  auto axis = softmax_param_->axis_;
  if (axis == -1) {
    softmax_param_->axis_ += n_dim;
    axis = softmax_param_->axis_;
  }
  auto in_shape = in_tensors_.front()->shape();
  int out_plane_size = 1;
  for (int i = 0; i < axis; ++i) {
    out_plane_size *= in_shape[i];
  }
  int in_plane_size = 1;
  for (int i = axis + 1; i < n_dim; i++) {
    in_plane_size *= in_shape[i];
  }

  sum_data_ =
    reinterpret_cast<float16_t *>(context_->allocator->Malloc(out_plane_size * in_plane_size * sizeof(float16_t)));
  if (sum_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc data for softmax fail!";
    return RET_ERROR;
  }
  memset(sum_data_, 0, out_plane_size * in_plane_size * sizeof(float16_t));

  input_fp16_ = ConvertInputFp32toFp16(in_tensors_.at(kInputIndex), context_);
  if (input_fp16_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_ERROR;
  }
  output_fp16_ = MallocOutputFp16(out_tensors_.at(kOutputIndex), context_);
  if (output_fp16_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_ERROR;
  }
  return RET_OK;
}

void SoftmaxFp16CPUKernel::FreeTmpBuffer() {
  if (sum_data_ != nullptr) {
    context_->allocator->Free(sum_data_);
    sum_data_ = nullptr;
  }
  if (in_tensors_.at(kInputIndex)->data_type() == kNumberTypeFloat32) {
    if (input_fp16_ != nullptr) {
      context_->allocator->Free(input_fp16_);
      input_fp16_ = nullptr;
    }
  }

  if (out_tensors_.at(kOutputIndex)->data_type() == kNumberTypeFloat32) {
    if (output_fp16_ != nullptr) {
      context_->allocator->Free(output_fp16_);
      output_fp16_ = nullptr;
    }
  }
}

int SoftmaxFp16CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return RET_ERROR;
  }
  ret = MallocTmpBuffer();
  if (ret != RET_OK) {
    FreeTmpBuffer();
    MS_LOG(ERROR) << "MallocTmpBuffer failed";
    return RET_ERROR;
  }
  SoftmaxFp16(input_fp16_, output_fp16_, sum_data_, softmax_param_);
  auto out_tensor = out_tensors_.at(kOutputIndex);
  if (out_tensor->data_type() == kNumberTypeFloat32) {
    Float16ToFloat32(output_fp16_, reinterpret_cast<float *>(out_tensor->Data()), out_tensor->ElementsNum());
  }
  FreeTmpBuffer();
  return RET_OK;
}

kernel::LiteKernel *CpuSoftmaxFp16KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                const std::vector<lite::tensor::Tensor *> &outputs,
                                                OpParameter *opParameter, const lite::Context *ctx,
                                                const kernel::KernelKey &desc,
                                                const mindspore::lite::PrimitiveC *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_SoftMax);
  auto *kernel = new (std::nothrow) SoftmaxFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SoftmaxFp16CPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_SoftMax, CpuSoftmaxFp16KernelCreator)

}  // namespace mindspore::kernel
