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
#include "src/runtime/kernel/arm/fp16/crop_fp16.h"

#include "include/errorcode.h"
#include "nnacl/crop_parameter.h"
#include "nnacl/fp16/cast_fp16.h"
#include "nnacl/fp16/crop_fp16.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/base/crop_base.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Crop;

namespace mindspore::kernel {

int CropFp16CPUKernel::Init() {
  auto ret = CropBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int CropFp16CPUKernel::ReSize() { return CropBaseCPUKernel::ReSize(); }

int CropFp16CPUKernel::DoExecute(int task_id) {
  Fp16Crop(input_ptr_, output_ptr_, task_id, crop_para_);
  return RET_OK;
}

static int CropFp16Run(void *cdata, int task_id) {
  auto g_kernel = reinterpret_cast<CropFp16CPUKernel *>(cdata);
  auto ret = g_kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "CropRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int CropFp16CPUKernel::Run() {
  input_ptr_ = ConvertInputFp32toFp16(in_tensors_.at(kInputIndex), context_);
  if (input_ptr_ == nullptr) {
    MS_LOG(ERROR) << "input or output is nullptr";
    return RET_ERROR;
  }

  output_ptr_ = MallocOutputFp16(out_tensors_.at(kOutputIndex), context_);
  if (output_ptr_ == nullptr) {
    FreeInputAndOutput();
    MS_LOG(ERROR) << "input or output is nullptr";
    return RET_ERROR;
  }

  auto ret = ParallelLaunch(this->context_->thread_pool_, CropFp16Run, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ParallelLaunch failed: " << ret;
    FreeInputAndOutput();
    return ret;
  }
  if (out_tensors_.at(kOutputIndex)->data_type() == kNumberTypeFloat32) {
    Float16ToFloat32(output_ptr_, reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->MutableData()),
                     out_tensors_.at(kOutputIndex)->ElementsNum());
  }
  FreeInputAndOutput();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Crop error error_code[" << ret << "]";
  }
  return ret;
}

void CropFp16CPUKernel::FreeInputAndOutput() {
  if (in_tensors_.at(kInputIndex)->data_type() == kNumberTypeFloat32) {
    context_->allocator->Free(input_ptr_);
    input_ptr_ = nullptr;
  }
  if (out_tensors_.at(kOutputIndex)->data_type() == kNumberTypeFloat32) {
    context_->allocator->Free(output_ptr_);
    output_ptr_ = nullptr;
  }
}

kernel::LiteKernel *CpuCropFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                             const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                             const InnerContext *ctx, const kernel::KernelKey &desc,
                                             const mindspore::lite::PrimitiveC *primitive) {
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Input opParameter is nullptr!";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Crop);
  auto *kernel = new (std::nothrow) CropFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new CropFp16CPUKernel fail!";
    free(opParameter);
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
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Crop, CpuCropFp16KernelCreator)
}  // namespace mindspore::kernel
