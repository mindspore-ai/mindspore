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
#include <algorithm>
#include "src/runtime/kernel/arm/fp32_grad/tuple_getitem.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_TupleGetItem;

namespace mindspore::kernel {

int TupleGetItemCPUKernel::Init() {
  if (in_tensors_.size() != 1) {
    MS_LOG(ERROR) << "Tuple Grad Filter should have one input";
    return RET_ERROR;
  }
  if (out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "Tuple Grad Filter should have one output";
    return RET_ERROR;
  }
  return RET_OK;
}

int TupleGetItemCPUKernel::ReSize() { return RET_OK; }

int TupleGetItemCPUKernel::Execute(int task_id) {
  auto in = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto out = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());

  size_t length = in_tensors_.at(0)->ElementsNum();

  size_t stride = UP_DIV(length, thread_count_);
  size_t count = MSMIN(stride, length - stride * task_id);

  size_t start = stride * task_id;
  size_t end = start + count;

  std::copy(&(in[start]), &(in[end]), &(out[start]));
  return RET_OK;
}

int TupleRun(void *cdata, int task_id) {
  auto tuple_kernel = reinterpret_cast<TupleGetItemCPUKernel *>(cdata);
  auto error_code = tuple_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "tuple grad error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int TupleGetItemCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, TupleRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "tuple function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuTupleGetItemFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                     const std::vector<lite::Tensor *> &outputs,
                                                     OpParameter *opParameter, const lite::InnerContext *ctx,
                                                     const kernel::KernelKey &desc, const lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_TupleGetItem);
  auto *kernel = new (std::nothrow) TupleGetItemCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new TupleGetItemCPUKernel failed!";
    free(opParameter);
    return nullptr;
  }

  auto ret = kernel->Init();
  if (RET_OK != ret) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_TupleGetItem, CpuTupleGetItemFp32KernelCreator)
}  // namespace mindspore::kernel
