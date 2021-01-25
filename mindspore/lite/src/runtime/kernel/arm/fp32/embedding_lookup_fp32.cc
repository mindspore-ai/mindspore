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

#include "src/runtime/kernel/arm/fp32/embedding_lookup_fp32.h"
#include "include/errorcode.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_EmbeddingLookupFusion;

namespace mindspore::kernel {
int EmbeddingLookupCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int EmbeddingLookupCPUKernel::ReSize() {
  param_->ids_size_ = in_tensors_.back()->ElementsNum();
  param_->layer_size_ = 1;
  auto in_shape = in_tensors_.front()->shape();
  for (size_t i = 1; i < in_shape.size(); ++i) {
    param_->layer_size_ *= in_shape[i];
  }

  param_->layer_num_ = 0;
  for (size_t i = 0; i < in_tensors_.size() - 1; ++i) {
    param_->layer_num_ += in_tensors_[i]->shape()[0];
  }
  return RET_OK;
}

int EmbeddingLookupCPUKernel::DoExcute(int task_id) {
  auto ids_addr = reinterpret_cast<int *>(in_tensors_.back()->MutableData());
  auto output_addr = reinterpret_cast<float *>(out_tensors_.front()->MutableData());
  int error_code = EmbeddingLookup(input_addr_, ids_addr, output_addr, param_, task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "embedding lookup error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int EmbeddingLookupRun(void *cdata, int task_id) {
  auto kernel = reinterpret_cast<EmbeddingLookupCPUKernel *>(cdata);
  auto ret = kernel->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "EmbeddingLookupRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int EmbeddingLookupCPUKernel::Run() {
  MS_ASSERT(context_->allocator != nullptr);
  input_addr_ =
    reinterpret_cast<float *>(context_->allocator->Malloc(sizeof(float) * param_->layer_size_ * param_->layer_num_));
  param_->is_regulated_ = reinterpret_cast<bool *>(context_->allocator->Malloc(sizeof(bool) * param_->layer_num_));
  if (input_addr_ == nullptr || param_->is_regulated_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    FreeRunBuff();
    return RET_ERROR;
  }
  for (int i = 0; i < param_->layer_num_; ++i) {
    param_->is_regulated_[i] = param_->max_norm_ == 0;
  }
  int dest_loc = 0;
  for (size_t i = 0; i < in_tensors_.size() - 1; i++) {
    auto input_t = reinterpret_cast<float *>(in_tensors_.at(i)->MutableData());
    memcpy(input_addr_ + dest_loc, input_t, sizeof(float) * in_tensors_.at(i)->ElementsNum());
    dest_loc += in_tensors_.at(i)->ElementsNum();
  }
  auto ret = ParallelLaunch(this->context_->thread_pool_, EmbeddingLookupRun, this, op_parameter_->thread_num_);
  FreeRunBuff();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "EmbeddingLookup error: error_code[" << ret << "]";
  }
  return ret;
}

void EmbeddingLookupCPUKernel::FreeRunBuff() {
  context_->allocator->Free(input_addr_);
  context_->allocator->Free(param_->is_regulated_);
  input_addr_ = nullptr;
  param_->is_regulated_ = nullptr;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_EmbeddingLookupFusion, LiteKernelCreator<EmbeddingLookupCPUKernel>)
}  // namespace mindspore::kernel
