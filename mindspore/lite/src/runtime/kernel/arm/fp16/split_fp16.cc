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
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/fp16/split_fp16.h"
#include "src/runtime/kernel/arm/fp16/split_fp16.h"
#include "src/runtime/kernel/arm/base/split_base.h"
#include "src/runtime/kernel/arm/nnacl/split.h"
#include "src/runtime/kernel/arm/nnacl/split_parameter.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Split;

namespace mindspore::kernel {

int SplitFp16CPUKernel::Init() {
  auto ret = SplitBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  output_ptr_.resize(param->num_split_);

  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int SplitFp16CPUKernel::ReSize() { return SplitBaseCPUKernel::ReSize(); }

int SplitFp16CPUKernel::Split(int task_id) {
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_n_stride_;
  auto ret = DoSplitFp16(input_ptr_, output_ptr_.data(), in_tensors_.front()->shape().data(), thread_offset,
                         num_unit_thread, param);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Split error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SplitRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<SplitFp16CPUKernel *>(cdata);
  auto ret = g_kernel->Split(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SplitRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SplitFp16CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return RET_ERROR;
  }
  auto in_tensor = in_tensors_.front();
  if (in_tensor->data_type() == kNumberTypeFloat32) {
    input_ptr_ =
      reinterpret_cast<float16_t *>(context_->allocator->Malloc(in_tensor->ElementsNum() * sizeof(float16_t)));
    Float32ToFloat16(reinterpret_cast<float *>(in_tensor->Data()), input_ptr_, in_tensor->ElementsNum());
  } else {
    input_ptr_ = reinterpret_cast<float16_t *>(in_tensor->Data());
  }
  for (int i = 0; i < param->num_split_; i++) {
    if (in_tensor->data_type() == kNumberTypeFloat32) {
      output_ptr_[i] = reinterpret_cast<float16_t *>(
        context_->allocator->Malloc(out_tensors_.at(i)->ElementsNum() * sizeof(float16_t)));
      Float32ToFloat16(reinterpret_cast<float *>(out_tensors_.at(i)->Data()), output_ptr_[i],
                       out_tensors_.at(i)->ElementsNum());
    } else {
      output_ptr_[i] = reinterpret_cast<float16_t *>(out_tensors_.at(i)->Data());
    }
  }
  ret = LiteBackendParallelLaunch(SplitRun, this, thread_n_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "split error error_code[" << ret << "]";
    return RET_ERROR;
  }
  if (in_tensor->data_type() == kNumberTypeFloat32) {
    context_->allocator->Free(input_ptr_);
    input_ptr_ = nullptr;
  }
  for (int i = 0; i < param->num_split_; i++) {
    if (in_tensor->data_type() == kNumberTypeFloat32) {
      context_->allocator->Free(output_ptr_[i]);
      output_ptr_[i] = nullptr;
    }
    return RET_OK;
  }
}

kernel::LiteKernel *CpuSplitFp16KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const Context *ctx,
                                              const kernel::KernelKey &desc,
                                              const mindspore::lite::PrimitiveC *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Split);
  auto *kernel = new (std::nothrow) SplitFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new SplitFp16CPUKernel fail!";
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
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Split, CpuSplitFp16KernelCreator)

}  // namespace mindspore::kernel
