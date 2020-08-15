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

#include "src/runtime/kernel/arm/fp32/expandDims.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_ExpandDims;

namespace mindspore::kernel {
int ExpandDimsCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ExpandDimsCPUKernel::ReSize() {
  data_size_ = in_tensors_.at(0)->ElementsNum();
  thread_sz_count_ = MSMIN(thread_count_, data_size_);
  thread_sz_stride_ = UP_DIV(data_size_, thread_sz_count_);
  return RET_OK;
}

int ExpandDimsCPUKernel::DoExpandDims(int task_id) {
  size_t size = MSMIN(thread_sz_stride_, data_size_ - task_id * thread_sz_stride_);
  if (size == 0) {
    return RET_OK;
  }
  int offset = task_id * thread_sz_stride_;
  int ret = ExpandDims(in_ptr_ + offset, out_ptr_ + offset, size * sizeof(float));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ExpandDimsRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ExpandDimsRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<ExpandDimsCPUKernel *>(cdata);
  auto ret = g_kernel->DoExpandDims(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ExpandDimsRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int ExpandDimsCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  in_ptr_ = reinterpret_cast<float *>(in_tensors_.at(0)->Data());
  out_ptr_ = reinterpret_cast<float *>(out_tensors_.at(0)->Data());
  auto ret = LiteBackendParallelLaunch(ExpandDimsRun, this, thread_sz_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ExpandDimsRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuExpandsDimsFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                    const std::vector<lite::tensor::Tensor *> &outputs,
                                                    OpParameter *opParameter, const lite::Context *ctx,
                                                    const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_ExpandDims);
  auto *kernel = new (std::nothrow) ExpandDimsCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new ExpandDimsCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_ExpandDims, CpuExpandsDimsFp32KernelCreator)
}  // namespace mindspore::kernel
