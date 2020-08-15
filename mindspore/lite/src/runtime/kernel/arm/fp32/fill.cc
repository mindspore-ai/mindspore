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

#include "src/runtime/kernel/arm/fp32/fill.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Fill;

namespace mindspore::kernel {

namespace {
constexpr int kInputNum = 1;
constexpr int kOutputNum = 1;
}  // namespace

int FillCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int FillCPUKernel::ReSize() {
  data_size_ = out_tensors_.front()->ElementsNum();
  thread_sz_count_ = MSMIN(thread_count_, data_size_);
  thread_sz_stride_ = UP_DIV(data_size_, thread_sz_count_);
  return RET_OK;
}

int FillCPUKernel::DoFill(int task_id) {
  int size = MSMIN(thread_sz_stride_, data_size_ - task_id * thread_sz_stride_);
  if (size <= 0) {
    return RET_OK;
  }
  int offset = task_id * thread_sz_stride_;
  int ret = Fill(out_ptr_ + offset, size, src_data_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FillRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int FillRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<FillCPUKernel *>(cdata);
  auto ret = g_kernel->DoFill(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FillRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int FillCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << prepare_ret;
    return prepare_ret;
  }
  auto fillData = in_tensors_.at(in_tensors_.size() - 1);
  auto output = out_tensors_.front();
  auto fill_data = reinterpret_cast<float *>(fillData->Data());
  src_data_ = fill_data[0];
  out_ptr_ = reinterpret_cast<float *>(output->Data());
  auto ret = LiteBackendParallelLaunch(FillRun, this, thread_sz_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "FillRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuFillFp32KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                             const std::vector<lite::tensor::Tensor *> &outputs,
                                             OpParameter *opParameter, const lite::Context *ctx,
                                             const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  if (opParameter == nullptr) {
    MS_LOG(ERROR) << "Create kernel failed, opParameter is nullptr, type: PrimitiveType_Fill. ";
    return nullptr;
  }
  MS_ASSERT(desc.type == schema::PrimitiveType_Fill);
  auto *kernel = new (std::nothrow) FillCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new FillCPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Fill, CpuFillFp32KernelCreator)
}  // namespace mindspore::kernel
